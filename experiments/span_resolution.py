"""Sealed post-score span-resolution deletion diagnostics; no model execution."""

from pathlib import Path

import numpy as np

from experiments import data, fullgap_dataset as fullgap, lumina_dataset
from experiments.pipeline import _complete_predictions, _validate_prediction, _verify_predictions
from experiments.provenance import completed_identity, research_identity
from experiments.research.sequence_controls import span_order_checks, span_score_components
from myproject.measure import token_scores


ORIGINALS = ('witness', 'lumina')
TRANSFORMS = ('span_mean', 'span_centered')
METHODS = tuple(f'{original}_{transform}' for original in ORIGINALS for transform in TRANSFORMS)
CONTRASTS = {f'{method}_minus_{original}': (original, method)
             for original in ORIGINALS for method in METHODS if method.startswith(original+'_')}
COLLECTION_MEMBERS = {'plan.json', 'complete.json', 'numeric_checks.json'} | {
    f'{method}/{name}' for method in METHODS for name in
    ('run.json', 'complete.json', 'predictions/seal.json')}


def _sources(native, reference_scores, external):
    fullgap._verify_simple_seal(native, expected={'plan.json', 'collection.json'})
    fullgap._verify_simple_seal(reference_scores, expected={'plan.json', 'complete.json'} | {
        f'{method}/{name}' for method in fullgap.ALLOWED_FULL_GAP_METHODS for name in
        ('run.json', 'complete.json', 'predictions/seal.json')})
    fullgap._verify_simple_seal(external, expected={
        'plan.json', 'collection.json', 'complete.json', 'fullgap_native/seal.json'} | {
        f'{method}/{name}' for method in lumina_dataset.METHODS for name in
        ('run.json', 'complete.json', 'predictions/seal.json')})
    parent = data.read_json(external/'plan.json')
    prepared = Path(parent['prepared'])
    manifest = data.read_json(prepared/'manifest.json')
    expected_bindings = {str(prepared/'inputs.jsonl'), str(prepared/'manifest.json'),
                         str(Path(parent['axis_reference_native'])/'seal.json'), parent['protocol'],
                         *manifest['raw_files'], *manifest.get('provenance_files', {})}
    if not isinstance(parent['input_bindings'], dict) or set(parent['input_bindings']) != expected_bindings:
        raise ValueError('LUMINA inference input binding member set changed')
    plan, sibling, native_plan, samples, records = lumina_dataset._read_collection(external)
    score_plan, score_samples, score_records, methods = fullgap._verify_scores(native, reference_scores)
    if (native != sibling or native_plan != score_plan or samples != score_samples
            or records != score_records or set(methods) != set(fullgap.ALLOWED_FULL_GAP_METHODS)):
        raise ValueError('Require the exact dense sibling and complete aligned seven-method scores')
    return plan, samples, records


def _bindings(native, reference_scores, external, protocol):
    return {str(path): data.file_hash(path) for path in
            (native/'seal.json', reference_scores/'seal.json', external/'seal.json', protocol)}


def _rows(native, reference_scores, external, samples, records):
    for index, (sample, record) in enumerate(zip(samples, records, strict=True)):
        raw = fullgap._load_observation(native/record['file'], sample, index)
        for original in ORIGINALS:
            directory = reference_scores if original == 'witness' else external
            prediction = data.read_json(data.prediction_path(directory/original/'predictions', sample.id))
            tokens = _validate_prediction(prediction, sample)
            scores = np.asarray([token['score'] for token in tokens], dtype=np.float64)
            expected = token_scores(raw, {'text': sample.response, 'offsets': raw['offsets']}, scores)
            if tokens != expected:
                raise ValueError('Original scores changed valid-token identity, offsets or coverage')
            components = span_score_components(scores, raw['answer_index'], raw['unit_index'])
            checks = span_order_checks(scores, components['span_centered'],
                                       raw['answer_index'], raw['unit_index'])
            for transform, values in components.items():
                yield sample, f'{original}_{transform}', {
                    **prediction, 'tokens': [{**token, 'score': float(value)}
                                             for token, value in zip(tokens, values, strict=True)]}, checks


def collect(native, reference_scores, external_collection, protocol, output):
    """Transform every valid token before reading any annotation or evaluation."""
    native, reference_scores, external, protocol, output = map(
        lambda value: Path(value).resolve(),
        (native, reference_scores, external_collection, protocol, output))
    if output.exists() or not protocol.is_file():
        raise ValueError('Require a new output and an existing span-resolution protocol')
    identity = research_identity()
    parent, samples, records = _sources(native, reference_scores, external)
    bindings = _bindings(native, reference_scores, external, protocol)
    plan = {'native': str(native), 'reference_scores': str(reference_scores),
            'external_collection': str(external), 'protocol': str(protocol),
            'input_bindings': bindings, 'methods': list(METHODS),
            'source': identity, 'annotation_reads': 0, 'new_model_calls': 0,
            'mean_population': 'all original valid scores, including later unknown labels'}
    output.mkdir(parents=True, exist_ok=False)
    data.write_json(output/'plan.json', plan)
    for method in METHODS:
        (output/method/'predictions').mkdir(parents=True)
    diagnostics = {original: {} for original in ORIGINALS}
    for sample, method, prediction, checks in _rows(native, reference_scores, external, samples, records):
        data.write_json(data.prediction_path(output/method/'predictions', sample.id), prediction)
        if method.endswith('_span_centered'):
            original = method.removesuffix('_span_centered')
            for name, value in checks.items():
                diagnostics[original][name] = diagnostics[original].get(name, 0)+value
    for method in METHODS:
        directory = output/method
        original, transform = method.split('_', 1)
        predictions = directory/'predictions'
        data.write_json(predictions/'seal.json', {path.name: data.file_hash(path)
                                                for path in sorted(predictions.iterdir())})
        data.write_json(directory/'run.json', {
            'input': parent['input'], 'input_sha256': parent['input_sha256'],
            'manifest_sha256': parent['manifest_sha256'], 'task': parent['task'],
            'method': method, 'original': original, 'transform': transform,
            'collection': str(output), 'source': identity, 'annotation_reads': 0,
            'claim_scope': 'post-score diagnostic deletion, not a new detector or calibrated belief',
        })
        _complete_predictions(directory, len(samples))
    data.write_json(output/'numeric_checks.json', diagnostics)
    _sources(native, reference_scores, external)
    fullgap._verify_paths(bindings, 'Span-resolution collection input changed')
    data.write_json(output/'complete.json', {'methods': list(METHODS), 'samples': len(samples),
                    'source': completed_identity(identity), 'annotation_reads': 0, 'new_model_calls': 0})
    names = ['plan.json', 'complete.json', 'numeric_checks.json',
             *(f'{method}/{name}' for method in METHODS for name in
               ('run.json', 'complete.json', 'predictions/seal.json'))]
    data.write_json(output/'seal.json', {name: data.file_hash(output/name) for name in names})
    return plan


def _read_collection(collection):
    fullgap._verify_simple_seal(collection, expected=COLLECTION_MEMBERS)
    plan = data.read_json(collection/'plan.json')
    native, scores, external, protocol = map(Path, (plan['native'], plan['reference_scores'],
                                                   plan['external_collection'], plan['protocol']))
    expected_bindings = {str(path) for path in
                         (native/'seal.json', scores/'seal.json', external/'seal.json', protocol)}
    if not isinstance(plan['input_bindings'], dict) or set(plan['input_bindings']) != expected_bindings:
        raise ValueError('Span-resolution input binding member set changed')
    fullgap._verify_paths(plan['input_bindings'], 'Span-resolution input changed')
    parent, samples, records = _sources(native, scores, external)
    if (plan['methods'] != list(METHODS) or plan['annotation_reads'] != 0
            or plan['new_model_calls'] != 0
            or plan['input_bindings'] != _bindings(native, scores, external, protocol)
            or data.read_json(collection/'complete.json') != {
                'methods': list(METHODS), 'samples': len(samples), 'source': plan['source'],
                'annotation_reads': 0, 'new_model_calls': 0}):
        raise ValueError('Span-resolution collection completion or provenance changed')
    for method in METHODS:
        directory = collection/method
        _verify_predictions(directory/'predictions')
        run, complete = data.read_json(directory/'run.json'), data.read_json(directory/'complete.json')
        if (complete != {'samples': len(samples), 'annotation_reads': 0,
                         'prediction_seal_sha256': data.file_hash(directory/'predictions/seal.json'),
                         'run_sha256': data.file_hash(directory/'run.json')}
                or any(run.get(key) != parent[key] for key in
                       ('input', 'input_sha256', 'manifest_sha256', 'task'))
                or run.get('method') != method or run.get('collection') != str(collection)
                or run.get('original') != method.split('_', 1)[0]
                or run.get('transform') != method.split('_', 1)[1]
                or run.get('source') != plan['source'] or run.get('annotation_reads') != 0):
            raise ValueError('Span-resolution prediction completion changed')
    diagnostics = {original: {} for original in ORIGINALS}
    for sample, method, expected, checks in _rows(native, scores, external, samples, records):
        if data.read_json(data.prediction_path(collection/method/'predictions', sample.id)) != expected:
            raise ValueError('Span-resolution scores differ from the declared all-token transform')
        if method.endswith('_span_centered'):
            for name, value in checks.items():
                original = method.removesuffix('_span_centered')
                diagnostics[original][name] = diagnostics[original].get(name, 0)+value
    if data.read_json(collection/'numeric_checks.json') != diagnostics:
        raise ValueError('Span-resolution numeric order checks changed')
    return plan, parent, native, scores, external, samples, records


def _checked_projection(evaluation, predictions, input_hash, annotation_hash, identities, samples):
    fullgap._verify_method_evaluation(evaluation, predictions, input_hash, annotation_hash)
    labels, units = fullgap._evaluation_rows(evaluation, identities)
    for sample in samples:
        tokens = data.read_json(data.prediction_path(predictions/'predictions', sample.id))['tokens']
        if any((sample.id, index) in units and units[sample.id, index]['score'] != token['score']
               for index, token in enumerate(tokens)):
            raise ValueError('Evaluated score differs from the sealed prediction')
    return labels, units


def _reference_evaluations(reference, external_evaluation, scores, external, parent, native,
                           samples, identities, *, read_rows=True):
    """Authenticate existing ten-method projections before any new annotations."""
    prepared = Path(parent['prepared'])
    manifest = data.read_json(prepared/'manifest.json')
    common = {'evaluation.json', 'evaluation_start.json', 'response_metadata.json'}
    fullgap._verify_simple_seal(reference, expected=common | {
        f'{method}/seal.json' for method in fullgap.ALLOWED_FULL_GAP_METHODS})
    fullgap._verify_simple_seal(external_evaluation, expected=common | {
        f'{method}/seal.json' for method in lumina_dataset.METHODS})
    baseline = data.read_json(reference/'evaluation.json')
    report = data.read_json(external_evaluation/'evaluation.json')
    if (baseline['score_seal_sha256'] != data.file_hash(scores/'seal.json')
            or baseline['native_seal_sha256'] != data.file_hash(native/'seal.json')
            or baseline['annotation_sha256'] != manifest['annotations_sha256']
            or baseline['samples_scored'] != len(samples)
            or baseline['dataset'] != manifest['dataset'] or baseline['task'] != parent['task']
            or baseline['bootstrap']['repetitions'] != fullgap.BOOTSTRAP_REPETITIONS
            or baseline['bootstrap']['seed'] != fullgap.BOOTSTRAP_SEED
            or set(baseline['methods']) != set(fullgap.ALLOWED_FULL_GAP_METHODS)
            or set(report['methods']) != set(fullgap.ALLOWED_FULL_GAP_METHODS) | set(lumina_dataset.METHODS)
            or report['dataset'] != manifest['dataset'] or report['task'] != parent['task']
            or report['samples_scored'] != len(samples)
            or report['bootstrap']['repetitions'] != fullgap.BOOTSTRAP_REPETITIONS
            or report['bootstrap']['seed'] != fullgap.BOOTSTRAP_SEED
            or report['reference_scores'] != str(scores) or report['reference_evaluation'] != str(reference)
            or report['input_bindings'].get(str(external/'seal.json')) != data.file_hash(external/'seal.json')
            or report['input_bindings'].get(str(reference/'seal.json')) != data.file_hash(reference/'seal.json')
            or report['input_bindings'].get(str(prepared/'annotations.jsonl')) != manifest['annotations_sha256']):
        raise ValueError('Existing dense seven/external ten evaluation binding changed')
    directories = {method: (reference/method, scores/method) for method in fullgap.ALLOWED_FULL_GAP_METHODS}
    directories.update({method: (external_evaluation/method, external/method)
                        for method in lumina_dataset.METHODS})
    # Authenticate every nested product before parsing any method's labels.
    for evaluation, predictions in directories.values():
        fullgap._verify_method_evaluation(evaluation, predictions, parent['input_sha256'],
                                         manifest['annotations_sha256'])
    labels, units = {}, {}
    for method, (evaluation, predictions) in directories.items():
        if read_rows:
            labels[method], units[method] = _checked_projection(
                evaluation, predictions, parent['input_sha256'], manifest['annotations_sha256'],
                identities, samples)
    if read_rows and any(rows != labels['witness'] for rows in labels.values()):
        raise ValueError('Existing reference methods changed label or unknown token identity')
    return manifest, directories, labels, units


def evaluate(collection, reference_evaluation, external_evaluation, output):
    """Evaluate fixed sealed deletions using existing label and bootstrap machinery."""
    from experiments.evaluate import evaluate_dataset

    collection, reference, external_evaluation, output = map(
        lambda value: Path(value).resolve(), (collection, reference_evaluation, external_evaluation, output))
    if output.exists():
        raise FileExistsError('Span-resolution evaluation output already exists')
    identity = research_identity()
    plan, parent, native, scores, external, samples, records = _read_collection(collection)
    identities = fullgap._native_token_identities(native, samples, records)
    manifest, directories, labels, units = _reference_evaluations(
        reference, external_evaluation, scores, external, parent, native, samples, identities)
    metadata, metadata_binding = fullgap._response_metadata(manifest, samples)
    native_units = fullgap._native_token_units(native, samples, records)
    bindings = {**plan['input_bindings'], **{str(path): data.file_hash(path) for path in
                (collection/'seal.json', reference/'seal.json', external_evaluation/'seal.json')},
                str(Path(parent['prepared'])/'annotations.jsonl'): manifest['annotations_sha256']}
    output.mkdir(parents=True, exist_ok=False)
    data.write_json(output/'evaluation_start.json', {'source': identity, 'input_bindings': bindings,
                    'methods': [*directories, *METHODS], 'contrasts': CONTRASTS})
    for method in METHODS:
        evaluate_dataset(collection/method, Path(parent['prepared'])/'annotations.jsonl', output/method,
                         repetitions=fullgap.BOOTSTRAP_REPETITIONS, seed=fullgap.BOOTSTRAP_SEED,
                         label_granularity='token')
        directories[method] = output/method, collection/method
        labels[method], units[method] = _checked_projection(
            output/method, collection/method, parent['input_sha256'], manifest['annotations_sha256'],
            identities, samples)
    gold = labels['witness']
    if any(rows != gold for rows in labels.values()):
        raise ValueError('Resolution/reference labels or unknown token identities differ')
    pooled, strata = fullgap._paired_views(list(identities), gold, units, metadata, native_units,
                                         contrasts=CONTRASTS)
    properties = {}
    for original in ORIGINALS:
        mean = pooled['paired_differences'][f'{original}_span_mean_minus_{original}']['within_span']
        centered = pooled['paired_differences'][f'{original}_span_centered_minus_{original}']['within_span']
        properties[original] = {'span_mean_within_span_auroc': mean['candidate'],
            'mean_property_satisfied': mean['candidate'] is None or mean['candidate'] == .5,
            'centered_within_span_auroc_difference': centered['difference'],
            'numeric_order_checks': data.read_json(collection/'numeric_checks.json')[original]}
    _read_collection(collection)
    _reference_evaluations(reference, external_evaluation, scores, external, parent, native,
                           samples, identities, read_rows=False)
    fullgap._verify_paths(bindings, 'Span-resolution evaluation input changed')
    if fullgap._response_metadata(manifest, samples) != (metadata, metadata_binding):
        raise ValueError('Released response strata changed during evaluation')
    summary = {'dataset': manifest['dataset'], 'task': parent['task'], 'samples_scored': len(samples),
        'methods': pooled['methods'], 'population': pooled['counts'], 'strata': strata,
        'paired_differences': pooled['paired_differences'], 'contrast_definitions': CONTRASTS,
        'properties': properties, 'input_bindings': bindings, 'metadata_binding': metadata_binding,
        'label_contract': parent['label_contract'], 'protocol_sha256': data.file_hash(Path(plan['protocol'])),
        'bootstrap': {'repetitions': fullgap.BOOTSTRAP_REPETITIONS, 'seed': fullgap.BOOTSTRAP_SEED,
                      'unit': 'source_id', 'cluster_semantics': manifest['bootstrap_type']},
        'unknown_policy': 'all valid scores enter means; unknown labels excluded only during evaluation',
        'factuality_calibration': False, 'evaluation_source': completed_identity(identity)}
    data.write_json(output/'evaluation.json', summary)
    data.write_json(output/'response_metadata.json', {'responses': metadata, 'binding': metadata_binding})
    names = ['evaluation_start.json', 'evaluation.json', 'response_metadata.json',
             *(f'{method}/seal.json' for method in METHODS)]
    data.write_json(output/'seal.json', {name: data.file_hash(output/name) for name in names})
    return summary
