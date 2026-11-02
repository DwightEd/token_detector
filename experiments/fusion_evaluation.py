"""Evaluate frozen rank-fusion scores without fitting or selecting a detector."""

from pathlib import Path
import re

import numpy as np

from . import data
from .datasets import _check_spans, read_jsonl, read_samples
from .evaluate import paired_source_differences, ranking, token_labels
from .provenance import completed_identity, research_identity
from .rank_fusion import AXES, RULE, _load_population, _text, _validate_axes


def _load_scores(path, run):
    """Read declared score/identity fields, leaving unrelated NPZ fields unopened."""
    with np.load(path, allow_pickle=False) as saved:
        result = {key: saved[key] for key in (*AXES, 'score', 'r4_score', 'witness_score')}
        for key in ['method', 'observer_id']:
            value = saved[key]
            if value.shape != ():
                raise ValueError('Scalar method/observer identity required')
            result[key] = _text(value).item()
    score = result['score']
    if score.ndim != 1 or not len(score):
        raise ValueError('Nonempty one-dimensional sealed scores required')
    for key in ['score', 'r4_score', 'witness_score']:
        value = result[key]
        if (value.shape != score.shape or value.dtype.kind not in 'fiu'
                or not np.isfinite(value).all()):
            raise ValueError('All three sealed score vectors must be finite and aligned')
    _validate_axes(result, score.shape)
    if (result['method'] != RULE['method'] or result['observer_id'] != run.get('observer_id')
            or run.get('tokens_scored') != len(score)
            or run.get('responses_scored') != len(set(result['sample_id'].tolist()))):
        raise ValueError('Sealed score method, observer or population counts differ')
    return result


def _align_samples(arrays, samples):
    """Retain the complete prepared order and every sealed original token."""
    ids = arrays['sample_id']
    starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
    stops = np.r_[starts[1:], len(ids)]
    if ids[starts].tolist() != [sample.id for sample in samples]:
        raise ValueError('Sealed sample IDs/order differ from complete prepared inputs')
    ranges = []
    for sample, start, stop in zip(samples, starts, stops, strict=True):
        selected = slice(start, stop)
        if (not isinstance(sample.response, str)
                or not np.all(arrays['source_id'][selected] == sample.source_id)
                or not np.all(arrays['response_sha256'][selected] == data.text_hash(sample.response))):
            raise ValueError('Prepared source/response identity differs from sealed scores')
        index = arrays['token_index'][selected]
        if (np.any(index[1:] <= index[:-1])
                or np.any(arrays['end'][selected] > len(sample.response))):
            raise ValueError('Original token order or response character bounds changed')
        ranges.append(selected)
    return ranges


def _labels(samples, ranges, arrays, annotations):
    """Apply released character spans only after the caller verifies the score seal."""
    rows = list(read_jsonl(annotations))
    by_id = {}
    for row in rows:
        key = row.get('id')
        if not isinstance(key, str) or not key or key in by_id:
            raise ValueError('Annotation IDs must be nonempty and unique')
        by_id[key] = row
    if set(by_id) != {sample.id for sample in samples}:
        raise ValueError('Annotations must cover the exact complete prepared sample IDs')
    labels = np.zeros(len(arrays['score']), dtype=bool)
    known = np.ones(len(labels), dtype=bool)
    for sample, selected in zip(samples, ranges, strict=True):
        row = by_id[sample.id]
        if row.get('response_sha256') != data.text_hash(sample.response):
            raise ValueError('Annotation response hash differs from prepared/sealed response')
        if row.get('granularity') != 'token' or row.get('coverage') != 'complete':
            raise ValueError('Only complete native token character-span annotations are supported')
        spans = row.get('spans')
        if (not isinstance(spans, list)
                or any(not isinstance(span, dict) or not {'start', 'end', 'label'} <= span.keys()
                       for span in spans)
                or any(span.get('label') is not None and
                       (type(span.get('label')) is not int or span['label'] not in (0, 1))
                       for span in spans)):
            raise ValueError('Character-span labels must be 0, 1 or explicitly undefined')
        _check_spans(sample.response, spans)
        offsets = np.column_stack((arrays['start'][selected], arrays['end'][selected]))
        response = {'offsets': offsets}
        positive = token_labels({'labels': [s for s in spans if s['label'] == 1]}, response)
        undefined = token_labels({'labels': [s for s in spans if s['label'] is None]}, response)
        labels[selected] = positive
        # As in the existing sentence projection, positive overlap takes precedence.
        # Unknowns affect evaluation only; original scoring population stays intact.
        known[selected] = positive | ~undefined
    if not known.any():
        raise ValueError('No known token labels remain for final evaluation')
    return labels, known


def evaluate(scores: Path, prepared: Path, output: Path,
             repetitions: int = 1000, seed: int = 17) -> dict:
    """Verify sealed scores, join released labels, and publish an immutable report.

    Every baseline and fusion score is already fixed. Confidence intervals resample
    whole source IDs; this exposed cohort remains exploratory, without selection.
    """
    if type(repetitions) is not int or repetitions < 1 or type(seed) is not int or seed < 0:
        raise ValueError('Positive bootstrap repetitions and a nonnegative integer seed required')
    scores, prepared, output = (Path(p).resolve() for p in (scores, prepared, output))
    base = (Path.cwd() / 'outputs').resolve()
    if output == base or not output.is_relative_to(base):
        raise ValueError('Evaluation outputs must be under current-project outputs/<run-id>')
    if output.exists():
        raise FileExistsError(output)

    expected = {'scores.npz', 'run.json'}
    data.verify_seal(scores, expected=expected)
    if {p.name for p in scores.iterdir()} != expected | {'seal.json'}:
        raise ValueError('Sealed score directory members changed')
    run = data.read_json(scores / 'run.json')
    if (run.get('status') != 'scores_sealed_not_evaluated' or run.get('rule') != RULE
            or run.get('annotation_reads') != 0 or run.get('metrics_computed') is not False
            or not isinstance(run.get('task'), str) or not run['task']):
        raise ValueError('Evaluation requires the frozen, unevaluated rank-fusion run')
    bindings = {}

    def bind(path, expected_hash=None):
        path = Path(path).resolve()
        digest = data.file_hash(path)
        if ((expected_hash is not None and digest != expected_hash)
                or (str(path) in bindings and bindings[str(path)] != digest)):
            raise ValueError('Frozen input binding changed')
        bindings[str(path)] = digest
        return path

    for name in ['scores.npz', 'run.json', 'seal.json']:
        bind(scores / name)
    original = run.get('input_sha256')
    if (not isinstance(original, dict) or not original
            or any(not isinstance(path, str) or not isinstance(digest, str)
                   or re.fullmatch('[0-9a-f]{64}', digest) is None
                   for path, digest in original.items())):
        raise ValueError('Frozen inference input bindings required')
    population_path = Path(run.get('population_manifest', '')).resolve()
    contract_root = population_path.parent
    contract = {str((contract_root / name).resolve()) for name in
                ['r4.npz', 'witness.npz', 'protocol.json', 'population.npz']}
    if (population_path.name != 'population.npz' or set(original) != contract
            or any(not Path(path).is_absolute() for path in original)):
        raise ValueError('Inference binding roster must be exactly the four canonical frozen inputs')
    annotation_path = (prepared / 'annotations.jsonl').resolve()
    if str(annotation_path) in contract:
        raise ValueError('Annotations cannot appear in inference bindings')
    protocol_path = bind(contract_root / 'protocol.json', original[str(contract_root / 'protocol.json')])
    protocol = data.read_json(protocol_path)
    frozen_fields = ['prepared_manifest_sha256', 'prepared_inputs_sha256',
                     'prepared_annotations_sha256']
    if (not isinstance(protocol, dict) or any(protocol.get(k) != v for k, v in RULE.items())
            or protocol.get('task') != run['task']
            or protocol.get('observer_id') != run.get('observer_id')
            or protocol.get('population_sha256') != run.get('population_sha256')
            or not isinstance(protocol.get('population_manifest'), str)
            or not protocol['population_manifest']
            or any(not isinstance(protocol.get(key), str)
                   or re.fullmatch('[0-9a-f]{64}', protocol[key]) is None for key in frozen_fields)):
        raise ValueError('Protocol must freeze the prepared manifest, inputs and annotations')
    declared_population = Path(protocol.get('population_manifest', ''))
    if not declared_population.is_absolute():
        declared_population = contract_root / declared_population
    if declared_population.resolve() != population_path:
        raise ValueError('Frozen population path differs from the inference contract')
    for path, digest in original.items():
        bind(path, digest)
    bind(population_path, run['population_sha256'])
    identity = research_identity()
    arrays = _load_scores(scores / 'scores.npz', run)
    population = _load_population(population_path)
    if any(not np.array_equal(arrays[key], population[key]) for key in AXES):
        raise ValueError('Sealed scores differ from the independent original-token population')
    manifest_path = bind(prepared / 'manifest.json', protocol['prepared_manifest_sha256'])
    manifest = data.read_json(manifest_path)
    if (manifest.get('inputs_sha256') != protocol['prepared_inputs_sha256']
            or manifest.get('annotations_sha256') != protocol['prepared_annotations_sha256']):
        raise ValueError('Prepared manifest differs from the frozen input/annotation bindings')
    inputs = bind(prepared / 'inputs.jsonl', protocol['prepared_inputs_sha256'])
    samples = list(read_samples(inputs))
    ranges = _align_samples(arrays, samples)

    # First annotation byte read occurs here, after the score seal, status,
    # complete population, inference bindings and response axes have passed.
    annotations = bind(prepared / 'annotations.jsonl', protocol['prepared_annotations_sha256'])
    labels, known = _labels(samples, ranges, arrays, annotations)
    y, groups = labels[known], arrays['source_id'][known]
    vectors = {'r4': arrays['r4_score'][known], 'witness': arrays['witness_score'][known],
               'fusion': arrays['score'][known]}
    methods = {name: ranking(y, values) for name, values in vectors.items()}
    differences = {}
    for baseline in ['r4', 'witness']:
        point = {key: (methods['fusion'][key] - methods[baseline][key]
                       if methods[baseline][key] is not None else None)
                 for key in ['auroc', 'ap']}
        differences['fusion_minus_' + baseline] = {
            'point': point,
            'bootstrap': paired_source_differences(
                y, vectors[baseline], vectors['fusion'], groups, repetitions, seed)}
    if any(data.file_hash(Path(path)) != digest for path, digest in bindings.items()):
        raise ValueError('Evaluation input changed while reading or computing metrics')
    data.verify_seal(scores, expected=expected)
    if {p.name for p in scores.iterdir()} != expected | {'seal.json'}:
        raise ValueError('Sealed score directory members changed')
    completed_identity(identity)
    report = {
        'method': RULE['method'], 'task': run['task'], 'observer_id': run['observer_id'],
        'scope': 'exploratory evaluation of previously exposed populations; fixed baselines, no selection',
        'fitting': False, 'threshold_selection': False,
        'annotation_first_read_after_score_seal': True,
        'label_provenance': manifest.get('dataset'),
        'source_grouping': manifest.get('source_grouping'),
        'bootstrap_type': manifest.get('bootstrap_type'),
        'bootstrap_grouping': {'field': 'source_id',
                               'prepared_group_type': manifest.get('bootstrap_type'),
                               'interpretation': 'released source IDs, not independently verified entities'},
        'population': {'responses_scored': len(samples),
                       'responses_evaluated': len(set(arrays['sample_id'][known].tolist())),
                       'tokens_scored': len(known), 'tokens_evaluated': int(known.sum()),
                       'unknown_tokens_excluded': int((~known).sum()),
                       'sources_evaluated': len(set(groups.tolist()))},
        'methods': methods, 'paired_differences': differences,
        'input_sha256': bindings, 'source': identity,
        'label_projection': 'positive character overlap; undefined-only overlaps excluded only from metrics',
        'alarm_budget': 'global top 5 percent of evaluated tokens, stable original-order tie breaking',
    }
    output.mkdir(parents=True, exist_ok=False)
    data.write_json(output / 'evaluation.json', report)
    data.write_json(output / 'seal.json', {'evaluation.json': data.file_hash(output / 'evaluation.json')})
    return report
