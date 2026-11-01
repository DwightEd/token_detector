"""Sealed original-axis LUMINA extraction and aligned external-baseline evaluation."""

from pathlib import Path
import time

import numpy as np

from experiments import data, fullgap_dataset as fullgap
from experiments.pipeline import _complete_predictions, _validate_prediction, _verify_predictions
from experiments.provenance import completed_identity, research_identity
from experiments.research.lumina import CORE, measure_group
from myproject.measure import token_scores


METHODS = ('lumina', 'lumina_ipr', 'lumina_negative_mmd')
FEATURES = ('lumina', 'ipr', 'mmd', 'present_logp', 'random_logp', 'absent_logp')
AXES = fullgap.OBSERVATION_FIELDS-{'logp'}
CONTRASTS = {f'{method}_minus_witness': ('witness', method) for method in METHODS}
DONOR_SEED = 17


def donor_mapping(groups):
    """A fixed other-example document per exact source/prompt group; no labels."""
    permutation = np.random.default_rng(DONOR_SEED).permutation(len(groups)).tolist()
    mapping = []
    for position, recipient in enumerate(permutation):
        (prompt, source), rows = groups[recipient]
        source_ids = {sample.source_id for _, sample in rows}
        if len(source_ids) != 1:
            raise ValueError('Each exact LUMINA group must have one original source ID')
        source_id = next(iter(source_ids))
        donor = None
        for step in range(1, len(groups)):
            candidate = permutation[(position+step) % len(groups)]
            (_, other), other_rows = groups[candidate]
            if (all(sample.source_id != source_id for _, sample in other_rows)
                    and data.text_hash(other) != data.text_hash(source)):
                donor = candidate
                break
        if donor is None:
            raise ValueError('No different-source-ID and different-source-text LUMINA donor exists')
        (_, other), other_rows = groups[donor]
        mapping.append({'recipient_group': recipient, 'donor_group': donor,
                        'recipient_source_id': source_id,
                        'donor_source_ids': sorted({sample.source_id for _, sample in other_rows}),
                        'recipient_ids': [sample.id for _, sample in rows],
                        'donor_ids': [sample.id for _, sample in other_rows],
                        'source_sha256': data.text_hash(source), 'donor_sha256': data.text_hash(other),
                        'prompt_sha256': data.text_hash(prompt)})
    return permutation, sorted(mapping, key=lambda row: row['recipient_group'])


def _feature_scores(raw, method):
    if method == 'lumina_negative_mmd':
        return -raw['mmd'][raw['target']]
    return raw['ipr' if method == 'lumina_ipr' else 'lumina'][raw['target']]


def _load_features(path, reference, sample, answer_index):
    axis = fullgap._load_observation(reference, sample, answer_index)
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != AXES | set(FEATURES):
            raise ValueError('LUMINA observation schema changed')
        raw = {name: saved[name] for name in saved.files}
    if any(raw[name].dtype != axis[name].dtype or not np.array_equal(raw[name], axis[name]) for name in AXES):
        raise ValueError('LUMINA changed an original token, offset, mask or span axis')
    if any(raw[name].shape != axis['answer_ids'].shape or not np.isfinite(raw[name]).all() for name in FEATURES):
        raise ValueError('LUMINA features must be finite on every original answer token')
    combined = (.5*raw['ipr'].astype(np.float32)-.5*raw['mmd'].astype(np.float32)).astype(np.float64)
    if not np.array_equal(raw['lumina'], combined):
        raise ValueError('LUMINA score no longer matches the released float32 component combination')
    return raw


def _write_predictions(output, plan, samples, records, reference_records, identity):
    native = Path(plan['reference_native'])
    for method in METHODS:
        directory = output/method
        predictions = directory/'predictions'
        predictions.mkdir(parents=True)
        for index, (sample, record, reference) in enumerate(zip(samples, records, reference_records, strict=True)):
            raw = _load_features(output/record['file'], native/reference['file'], sample, index)
            tokens = token_scores(raw, {'text': sample.response, 'offsets': raw['offsets']},
                                  _feature_scores(raw, method))
            row = {'id': sample.id, 'source_id': sample.source_id,
                   'response_sha256': data.text_hash(sample.response), 'tokens': tokens}
            _validate_prediction(row, sample)
            data.write_json(data.prediction_path(predictions, sample.id), row)
        data.write_json(predictions/'seal.json', {path.name: data.file_hash(path)
                                                for path in sorted(predictions.iterdir())})
        data.write_json(directory/'run.json', {
            'input': plan['input'], 'input_sha256': plan['input_sha256'],
            'manifest_sha256': plan['manifest_sha256'], 'task': plan['task'], 'method': method,
            'collection': str(output), 'core': CORE, 'source': identity, 'annotation_reads': 0,
        })
        _complete_predictions(directory, len(samples))


def collect(prepared, model, reference_native, protocol, output, device='cuda:0'):
    """Extract dense S/R/A features and a matched S/A sibling before labels."""
    import torch
    from myproject.api import HallucinationDetector

    prepared, model, reference_native, protocol, output = map(
        lambda value: Path(value).resolve(), (prepared, model, reference_native, protocol, output))
    if output.exists() or not protocol.is_file():
        raise ValueError('Require a new output and an existing frozen LUMINA protocol')
    prepared, manifest, samples, groups, bindings = fullgap._prepared(prepared)
    reference_plan, reference_samples, reference_records = fullgap._read_native(reference_native)
    model_bindings = fullgap._hash_files(fullgap._model_files(model))
    if (reference_plan['prepared'] != str(prepared) or reference_samples != samples
            or reference_plan['model'] != str(model) or reference_plan['model_files'] != model_bindings):
        raise ValueError('LUMINA must use the exact reference input population and observer weights')
    permutation, mapping = donor_mapping(groups)
    identity = research_identity()
    bindings.update({str(reference_native/'seal.json'): data.file_hash(reference_native/'seal.json'),
                     str(protocol): data.file_hash(protocol)})
    axis_reference_native = reference_native
    reference_native = output/'fullgap_native'
    plan = {
        'prepared': str(prepared), 'input': str(prepared/'inputs.jsonl'),
        'input_sha256': manifest['inputs_sha256'], 'manifest_sha256': data.file_hash(prepared/'manifest.json'),
        'dataset': manifest['dataset'], 'task': samples[0].task,
        'label_contract': fullgap._label_contract(manifest['dataset']),
        'axis_reference_native': str(axis_reference_native),
        'reference_native': str(reference_native), 'input_bindings': bindings,
        'model': str(model), 'model_files': model_bindings, 'device': device,
        'protocol': str(protocol), 'protocol_sha256': bindings[str(protocol)],
        'source': identity, 'core': CORE, 'methods': list(METHODS),
        'worlds': ['original_source', 'other_example_source', 'source_absent'], 'backend': 'matched_sdpa',
        'backbone_execution': 'dense_full_prefix', 'prefix_cache_reuse': False,
        'donor_seed': DONOR_SEED, 'donor_permutation': permutation, 'donors': mapping,
        'population': {'responses': len(samples), 'prefix_groups': len(groups)},
        'observation_adapter': {'original_response_axis': True, 'prepend_answer_space': False,
                                'truncation': False, 'full_answer_prefix': True},
        'annotation_reads': 0, 'new_generation_calls': 0, 'judge_calls': 0,
    }
    output.mkdir(parents=True, exist_ok=False)
    data.write_json(output/'plan.json', plan)
    (output/'observations').mkdir()
    (reference_native/'observations').mkdir(parents=True)
    native_plan = {
        'prepared': str(prepared), 'input': plan['input'], 'input_sha256': plan['input_sha256'],
        'manifest_sha256': plan['manifest_sha256'], 'raw_files': manifest['raw_files'],
        'provenance_files': manifest.get('provenance_files', {}),
        'annotation_sha256_from_manifest': manifest['annotations_sha256'],
        'model': str(model), 'model_files': model_bindings,
        'protocol': str(protocol), 'protocol_sha256': plan['protocol_sha256'],
        'inference_bindings': dict(bindings),
        'source': identity, 'task': samples[0].task, 'dataset': manifest['dataset'],
        'label_contract': plan['label_contract'], 'device': device,
        'backend': 'matched_sdpa', 'backbone_execution': 'dense_full_prefix',
        'prefix_cache_reuse': False, 'world_columns': ['source_present', 'source_absent'],
        'score_gap': 'source_absent_logp - source_present_logp',
        'parent_collection': str(output), 'parent_plan_sha256': data.file_hash(output/'plan.json'),
        'source_present_observation': 'same actual dense S forward as parent LUMINA; exact saved logp bytes',
        'axis_reference_native': str(axis_reference_native),
        'annotation_reads': 0, 'claim_scope': 'matched dense source observations, not factuality calibration',
    }
    data.write_json(reference_native/'plan.json', native_plan)
    torch.manual_seed(42)
    holder = HallucinationDetector.from_pretrained(model, task=samples[0].task, method='witness', device=device)
    counts = {'backbone_calls': 0, 'processed_tokens': 0}

    def count(_module, _args, kwargs):
        counts['backbone_calls'] += 1
        counts['processed_tokens'] += kwargs['input_ids'].numel()

    handle = holder.model.model.register_forward_pre_hook(count, with_kwargs=True)
    cuda = holder.model.device.type == 'cuda'
    if cuda:
        torch.cuda.reset_peak_memory_stats(holder.model.device)
        torch.cuda.synchronize(holder.model.device)
    started = time.monotonic()
    records, native_records, reference_logp_error = [], [], 0.
    try:
        for group_index, ((prompt, source), rows) in enumerate(groups):
            donor = groups[mapping[group_index]['donor_group']][0][1]
            measured, encoded = measure_group(holder.model, holder.tokenizer, prompt, source,
                                              donor, [sample.response for _, sample in rows])
            if len(measured) != len(rows) or len(encoded) != len(rows):
                raise ValueError('LUMINA measurement dropped an original response')
            for (index, sample), features, encoding in zip(rows, measured, encoded, strict=True):
                reference = reference_records[index]
                axis = fullgap._load_observation(axis_reference_native/reference['file'], sample, index)
                current = {**fullgap._axis_arrays(encoding, index),
                           'answer_ids': np.asarray(encoding['answer_ids']),
                           'offsets': np.asarray(encoding['offsets'], dtype=np.int64).reshape(-1, 2),
                           'special_ids': np.asarray(encoding['special_ids'])}
                if (encoding['text'] != sample.response or set(features) != set(FEATURES)
                        or any(not np.array_equal(current[name], axis[name]) for name in AXES)):
                    raise ValueError('LUMINA adapter changed original response identity or axes')
                file = data.prediction_path(output/'observations', sample.id).with_suffix('.npz')
                with file.open('xb') as stream:
                    np.savez_compressed(stream, **{name: axis[name] for name in AXES}, **features)
                _load_features(file, axis_reference_native/reference['file'], sample, index)
                reference_logp_error = max(reference_logp_error, float(np.max(
                    np.abs(features['present_logp']-axis['logp'][:, 0]))))
                records.append({'id': sample.id, 'source_id': sample.source_id, 'answer_index': index,
                                'prefix_group': group_index, 'file': str(file.relative_to(output)),
                                'sha256': data.file_hash(file), 'response_sha256': data.text_hash(sample.response)})
                native_file = data.prediction_path(reference_native/'observations', sample.id).with_suffix('.npz')
                with native_file.open('xb') as stream:
                    np.savez_compressed(stream, **{name: axis[name] for name in AXES},
                                        logp=np.column_stack((features['present_logp'], features['absent_logp'])))
                native_records.append({**records[-1], 'file': str(native_file.relative_to(reference_native)),
                                       'sha256': data.file_hash(native_file)})
            print('LUMINA_COLLECT', group_index+1, len(groups), flush=True)
    finally:
        handle.remove()
    if cuda:
        torch.cuda.synchronize(holder.model.device)
    records.sort(key=lambda row: row['answer_index'])
    native_records.sort(key=lambda row: row['answer_index'])
    if [record['answer_index'] for record in records] != list(range(len(samples))):
        raise ValueError('LUMINA complete original response coverage changed')
    if counts['backbone_calls'] != 3*len(samples):
        raise ValueError('Dense LUMINA requires exactly three actual backbone calls per response')
    fullgap._verify_paths(bindings, 'LUMINA inference inputs changed')
    if fullgap._hash_files(fullgap._model_files(model)) != model_bindings:
        raise ValueError('LUMINA observer model bytes or runtime file set changed')
    fullgap._read_native(axis_reference_native)
    elapsed = time.monotonic()-started
    data.write_json(reference_native/'collection.json', {
        'records': native_records, 'source': completed_identity(identity), 'annotation_reads': 0,
        'counts': {'source_present_dense_forwards': len(samples), 'source_absent_dense_forwards': len(samples),
                   'random_dense_forwards_in_parent_only': len(samples)},
        'parent_backbone_counts': counts, 'parent_extraction_seconds': elapsed,
    })
    data.write_json(reference_native/'seal.json', {name: data.file_hash(reference_native/name)
                                                for name in ('plan.json', 'collection.json')})
    data.write_json(output/'collection.json', {
        'records': records, 'counts': {**counts, 'dense_forwards_per_response': 3, 'prefix_cache_reuse': False},
        'seconds': elapsed,
        'peak_cuda_bytes': torch.cuda.max_memory_allocated(holder.model.device) if cuda else None,
        'cached_axis_reference_present_logp_max_abs_difference': reference_logp_error,
        'logp_difference_scope': 'descriptive dense versus cached axis reference; not numerical equivalence or calibration',
        'model_dtype': str(holder.model.dtype), 'source': completed_identity(identity), 'annotation_reads': 0,
    })
    _write_predictions(output, plan, samples, records, native_records, identity)
    data.write_json(output/'complete.json', {'methods': list(METHODS), 'samples': len(samples),
                                         'source': completed_identity(identity), 'annotation_reads': 0})
    names = ['plan.json', 'collection.json', 'complete.json', 'fullgap_native/seal.json',
             *(f'{method}/{name}' for method in METHODS for name in
               ('run.json', 'complete.json', 'predictions/seal.json'))]
    data.write_json(output/'seal.json', {name: data.file_hash(output/name) for name in names})


def _read_collection(output):
    fullgap._verify_simple_seal(output)
    plan = data.read_json(output/'plan.json')
    fullgap._verify_paths(plan['input_bindings'], 'Bound LUMINA inputs or protocol changed')
    native = Path(plan['reference_native'])
    native_plan, samples, reference = fullgap._read_native(native)
    if (native_plan.get('protocol') != plan['protocol']
            or native_plan.get('protocol_sha256') != plan['protocol_sha256']
            or native_plan.get('inference_bindings') != plan['input_bindings']
            or native_plan.get('input') != plan['input']
            or plan['input'] != str(Path(plan['prepared'])/'inputs.jsonl')):
        raise ValueError('Dense sibling inference protocol, input path or bindings differ from parent LUMINA')
    fullgap._verify_paths(native_plan['inference_bindings'], 'Dense sibling inference inputs changed')
    axis_native = Path(plan['axis_reference_native'])
    axis_plan, axis_samples, axis_records = fullgap._read_native(axis_native)
    _, _, prepared_samples, groups, _ = fullgap._prepared(Path(plan['prepared']))
    permutation, donors = donor_mapping(groups)
    if (prepared_samples != samples or axis_samples != samples
            or plan['prepared'] != native_plan['prepared']
            or plan['task'] != native_plan['task'] or plan['dataset'] != native_plan['dataset']
            or plan['model'] != native_plan['model'] or plan['annotation_reads'] != 0
            or plan['manifest_sha256'] != native_plan['manifest_sha256']
            or plan['protocol_sha256'] != data.file_hash(Path(plan['protocol']))
            or plan['core'] != CORE or plan['methods'] != list(METHODS)
            or plan['donor_permutation'] != permutation or plan['donors'] != donors
            or plan['model_files'] != native_plan['model_files']
            or plan['model_files'] != axis_plan['model_files']
            or plan.get('backbone_execution') != 'dense_full_prefix'
            or plan.get('prefix_cache_reuse') is not False
            or native_plan.get('source') != plan['source']
            or native_plan.get('axis_reference_native') != str(axis_native)
            or native_plan.get('backbone_execution') != 'dense_full_prefix'
            or native_plan.get('prefix_cache_reuse') is not False
            or native_plan.get('parent_collection') != str(output)
            or native_plan.get('parent_plan_sha256') != data.file_hash(output/'plan.json')
            or native != output/'fullgap_native'
            or plan['input_sha256'] != native_plan['input_sha256']
            or data.read_json(output/'complete.json') != {
                'methods': list(METHODS), 'samples': len(samples),
                'source': plan['source'], 'annotation_reads': 0}):
        raise ValueError('LUMINA core, donor or population provenance changed')
    records = data.read_json(output/'collection.json')['records']
    if len(records) != len(samples):
        raise ValueError('LUMINA observation coverage changed')
    for index, (sample, record, original, axis_record) in enumerate(
            zip(samples, records, reference, axis_records, strict=True)):
        file = output/record['file']
        if (record['id'] != sample.id or record['source_id'] != sample.source_id
                or record['answer_index'] != index or data.file_hash(file) != record['sha256']
                or record['response_sha256'] != data.text_hash(sample.response)):
            raise ValueError('LUMINA observation identity or bytes changed')
        raw = _load_features(file, axis_native/axis_record['file'], sample, index)
        dense = fullgap._load_observation(native/original['file'], sample, index)
        if any(raw[name].dtype != dense[name].dtype or raw[name].tobytes() != dense[name].tobytes() for name in AXES):
            raise ValueError('Dense source-gap sibling changed an original token axis')
        for column, name in enumerate(('present_logp', 'absent_logp')):
            if raw[name].dtype != dense['logp'].dtype or raw[name].tobytes() != dense['logp'][:, column].tobytes():
                raise ValueError('Dense sibling S/A columns differ from the actual LUMINA forwards')
    if {path.name for path in (output/'observations').iterdir()} != {Path(row['file']).name for row in records}:
        raise ValueError('LUMINA observation file set changed')
    for method in METHODS:
        directory = output/method
        _verify_predictions(directory/'predictions')
        run, complete = data.read_json(directory/'run.json'), data.read_json(directory/'complete.json')
        if (run != {'input': plan['input'], 'input_sha256': plan['input_sha256'],
                    'manifest_sha256': plan['manifest_sha256'], 'task': plan['task'], 'method': method,
                    'collection': str(output), 'core': CORE, 'source': plan['source'], 'annotation_reads': 0}
                or complete != {'samples': len(samples),
                    'prediction_seal_sha256': data.file_hash(directory/'predictions/seal.json'),
                    'run_sha256': data.file_hash(directory/'run.json'), 'annotation_reads': 0}):
            raise ValueError('LUMINA prediction completion binding changed')
        for index, (sample, record, original) in enumerate(zip(samples, records, reference, strict=True)):
            prediction = data.read_json(data.prediction_path(directory/'predictions', sample.id))
            tokens = _validate_prediction(prediction, sample)
            raw = _load_features(output/record['file'], native/original['file'], sample, index)
            expected = token_scores(raw, {'text': sample.response, 'offsets': raw['offsets']},
                                    _feature_scores(raw, method))
            if tokens != expected:
                raise ValueError('LUMINA prediction differs from measured features or original token axes')
    return plan, native, native_plan, samples, reference


def evaluate(collection, reference_scores, reference_evaluation, output):
    """Authenticate the baseline and full7 reference before common label projection."""
    from experiments.evaluate import evaluate_dataset

    collection, reference_scores, reference_evaluation, output = map(
        lambda value: Path(value).resolve(), (collection, reference_scores, reference_evaluation, output))
    if output.exists():
        raise FileExistsError('LUMINA evaluation output already exists')
    identity = research_identity()
    plan, native, native_plan, samples, records = _read_collection(collection)
    if Path(data.read_json(reference_scores/'plan.json')['native']).resolve() != native:
        raise ValueError('LUMINA requires its own matched dense sibling, not cached-reference scores')
    _, reference_samples, _, reference_methods = fullgap._verify_scores(native, reference_scores)
    if reference_samples != samples or set(reference_methods) != set(fullgap.ALLOWED_FULL_GAP_METHODS):
        raise ValueError('Require the complete aligned reference seven-method packet')
    fullgap._verify_simple_seal(reference_evaluation)
    reference_report = data.read_json(reference_evaluation/'evaluation.json')
    prepared = Path(plan['prepared'])
    manifest = data.read_json(prepared/'manifest.json')
    if (reference_report['score_seal_sha256'] != data.file_hash(reference_scores/'seal.json')
            or reference_report['native_seal_sha256'] != data.file_hash(native/'seal.json')
            or reference_report['annotation_sha256'] != manifest['annotations_sha256']
            or reference_report['dataset'] != manifest['dataset']
            or reference_report['task'] != samples[0].task
            or reference_report['samples_scored'] != len(samples)
            or set(reference_report['methods']) != set(reference_methods)):
        raise ValueError('Reference evaluation score/native/annotation binding changed')
    for method in reference_methods:
        fullgap._verify_method_evaluation(reference_evaluation/method, reference_scores/method,
                                         plan['input_sha256'], manifest['annotations_sha256'])
    metadata, metadata_binding = fullgap._response_metadata(manifest, samples)
    identities = fullgap._native_token_identities(native, samples, records)
    native_units = fullgap._native_token_units(native, samples, records)
    bindings = {str(collection/'seal.json'): data.file_hash(collection/'seal.json'),
                str(reference_scores/'seal.json'): data.file_hash(reference_scores/'seal.json'),
                str(reference_evaluation/'seal.json'): data.file_hash(reference_evaluation/'seal.json'),
                str(prepared/'annotations.jsonl'): manifest['annotations_sha256']}
    output.mkdir(parents=True, exist_ok=False)
    data.write_json(output/'evaluation_start.json', {'source': identity, 'input_bindings': bindings,
                    'methods': [*reference_methods, *METHODS], 'contrasts': CONTRASTS,
                    'label_contract': plan['label_contract'], 'protocol_sha256': plan['protocol_sha256']})
    data.write_json(output/'response_metadata.json', {'responses': metadata, 'binding': metadata_binding})
    labels, units = {}, {}
    for method in [*reference_methods, *METHODS]:
        if method in METHODS:
            evaluate_dataset(collection/method, prepared/'annotations.jsonl', output/method,
                             repetitions=fullgap.BOOTSTRAP_REPETITIONS, seed=fullgap.BOOTSTRAP_SEED,
                             label_granularity='token')
            evaluation, scores = output/method, collection/method
        else:
            evaluation, scores = reference_evaluation/method, reference_scores/method
        fullgap._verify_method_evaluation(evaluation, scores, plan['input_sha256'], manifest['annotations_sha256'])
        labels[method], units[method] = fullgap._evaluation_rows(evaluation, identities)
        for sample in samples:
            prediction = data.read_json(data.prediction_path(scores/'predictions', sample.id))['tokens']
            if any((sample.id, index) in units[method]
                   and units[method][sample.id, index]['score'] != token['score']
                   for index, token in enumerate(prediction)):
                raise ValueError('Evaluation score differs from its sealed original-token prediction')
    gold = labels['witness']
    if any(labels[method] != gold for method in labels):
        raise ValueError('LUMINA/reference token labels or unknown identity differ')
    pooled, strata = fullgap._paired_views(list(identities), gold, units, metadata, native_units,
                                         contrasts=CONTRASTS)
    _read_collection(collection)
    fullgap._verify_scores(native, reference_scores)
    fullgap._verify_paths(bindings, 'LUMINA evaluation input binding changed')
    if fullgap._response_metadata(manifest, samples) != (metadata, metadata_binding):
        raise ValueError('LUMINA released metadata changed during evaluation')
    summary = {
        'dataset': manifest['dataset'], 'task': samples[0].task, 'samples_scored': len(samples),
        'methods': pooled['methods'], 'population': pooled['counts'],
        'paired_differences': pooled['paired_differences'], 'strata': strata,
        'contrast_definitions': CONTRASTS, 'core': CORE, 'label_contract': plan['label_contract'],
        'metadata_binding': metadata_binding, 'input_bindings': bindings,
        'reference_scores': str(reference_scores), 'reference_evaluation': str(reference_evaluation),
        'unknown_policy': 'undefined, uncovered and unalignable labels retained as unknown then excluded',
        'bootstrap': {'repetitions': fullgap.BOOTSTRAP_REPETITIONS, 'seed': fullgap.BOOTSTRAP_SEED,
                      'unit': 'source_id', 'cluster_semantics': manifest['bootstrap_type']},
        'protocol_sha256': plan['protocol_sha256'], 'factuality_calibration': False,
        'evaluation_source': completed_identity(identity),
    }
    data.write_json(output/'evaluation.json', summary)
    names = ['evaluation_start.json', 'evaluation.json', 'response_metadata.json',
             *(f'{method}/seal.json' for method in METHODS)]
    data.write_json(output/'seal.json', {name: data.file_hash(output/name) for name in names})
    return summary
