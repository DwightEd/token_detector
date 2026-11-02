"""Export sealed historical R4/V3 caches without opening annotations or weights."""

import json
from pathlib import Path, PurePosixPath
import re

import numpy as np

from . import data
from .datasets import read_samples
from .provenance import completed_identity, research_identity
from .rank_fusion import AXES, RULE, _validate_axes


def _member(root, name):
    if (not isinstance(name, str) or not name or '\\' in name
            or PurePosixPath(name).is_absolute()
            or any(part in {'.', '..'} for part in name.split('/'))):
        raise ValueError('Unsafe cache record member')
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Cache record escapes artifact root')
    return path


def _records(value):
    rows = value.get('records') if isinstance(value, dict) else None
    if (not isinstance(rows, list) or not rows
            or any(not isinstance(r, dict) or not isinstance(r.get('id'), str)
                   or not r['id'] or type(r.get('group')) is not int
                   or r['group'] < 0 or not isinstance(r.get('file'), str) or not r['file']
                   or any(not isinstance(r.get(key), str)
                          or re.fullmatch('[0-9a-f]{64}', r[key]) is None
                          for key in ['sha256', 'response_sha256']) for r in rows)
            or len({r['id'] for r in rows}) != len(rows)
            or len({r['file'] for r in rows}) != len(rows)):
        raise ValueError('Nonempty unique cache record identities required')
    return rows


def _observer(plan):
    files = plan.get('model_files_sha256', plan.get('model_files'))
    if not isinstance(files, dict) or not files:
        raise ValueError('Native model file digests required')
    normalized = {}
    for name, digest in files.items():
        if (not isinstance(name, str) or not name or not isinstance(digest, str)
                or re.fullmatch('[0-9a-f]{64}', digest) is None):
            raise ValueError('Invalid native model file digest')
        basename = name.replace('\\', '/').rsplit('/', 1)[-1]
        if not basename or basename in normalized:
            raise ValueError('Ambiguous native model file basename')
        normalized[basename] = digest
    return 'model-files-sha256:' + data.text_hash(json.dumps(normalized, sort_keys=True))


def _native_axis(path, sample, ordinal):
    # Explicit field access leaves unrelated object/annotation members unopened.
    with np.load(path, allow_pickle=False) as saved:
        raw = {k: saved[k] for k in
               ['full_token_id', 'offsets', 'special_ids', 'target', 'token_id', 'answer_index']}
    ids, offsets, target = raw['full_token_id'], raw['offsets'], raw['target']
    if (ids.ndim != 1 or ids.dtype.kind not in 'iu' or np.any(ids < 0)
            or offsets.shape != (len(ids), 2) or offsets.dtype.kind not in 'iu'
            or np.any(offsets < 0) or np.any(offsets > len(sample.response))
            or np.any(offsets[:, 1] < offsets[:, 0])
            or raw['special_ids'].ndim != 1 or raw['special_ids'].dtype.kind not in 'iu'):
        raise ValueError('Invalid independent native full token axis')
    wanted = np.flatnonzero((offsets[:, 1] > offsets[:, 0])
                           & ~np.isin(ids, raw['special_ids']))
    if (target.dtype.kind not in 'iu' or not np.array_equal(target, wanted)
            or raw['token_id'].dtype.kind not in 'iu'
            or not np.array_equal(raw['token_id'], ids[wanted])
            or raw['answer_index'].shape != wanted.shape
            or raw['answer_index'].dtype.kind not in 'iu'
            or not np.all(raw['answer_index'] == ordinal)):
        raise ValueError('Independent native valid token population changed')
    count = len(wanted)
    result = {'sample_id': np.repeat(sample.id, count),
              'source_id': np.repeat(sample.source_id, count),
              'response_sha256': np.repeat(data.text_hash(sample.response), count),
              'token_index': wanted, 'token_id': ids[wanted],
              'start': offsets[wanted, 0], 'end': offsets[wanted, 1]}
    _validate_axes(result, wanted.shape)
    return result, raw['answer_index'], offsets


def export(scores, native, prepared, group, output, taskname):
    """Verify the complete native TEST population, then export one metadata group.

    ``prepared`` supplies relocated inference folders in native-plan group order.
    Raw acquisition files and annotations are deliberately not opened or verified.
    Original plans, seals and file digests remain provenance, not a new LM replay.
    """
    scores, native, output = (Path(p).resolve() for p in (scores, native, output))
    prepared = [Path(p).resolve() for p in prepared]
    base = (Path.cwd() / 'outputs').resolve()
    if output == base or not output.is_relative_to(base):
        raise ValueError('Export outputs must be under current-project outputs/<run-id>')
    if output.exists():
        raise FileExistsError(output)
    if not isinstance(taskname, str) or not taskname.strip():
        raise ValueError('Declare a nonempty task name')
    identity = research_identity()
    bindings = {}

    def bind(path, expected=None):
        digest = data.file_hash(path)
        if expected is not None and digest != expected:
            raise ValueError('Historical cache or prepared input digest changed')
        bindings[str(path)] = digest
        return digest

    native_members = data.verify_seal(native, expected={'plan.json', 'collection.json'})
    native_seal = bind(native / 'seal.json')
    plan = data.read_json(native / 'plan.json')
    collection = data.read_json(native / 'collection.json')
    native_records = _records(collection)
    for name in ['plan.json', 'collection.json']:
        bind(native / name, native_members[name])
    # Inspect the record roster to declare the exact member set before hashing
    # any seal-listed file; an unexpected annotation member must never be read.
    preliminary = _records(data.read_json(scores / 'complete.json'))
    score_seal = data.verify_seal(scores, expected={'plan.json', 'complete.json'}
                                  | {r['file'] for r in preliminary})
    score_plan = data.read_json(scores / 'plan.json')
    score_records = _records(data.read_json(scores / 'complete.json'))
    if score_records != preliminary:
        raise ValueError('Score roster changed during seal verification')
    for name, digest in score_seal.items():
        bind(_member(scores, name), digest)
    bind(scores / 'seal.json')
    if (score_plan.get('native_seal_sha256') != native_seal
            or score_plan.get('method') != 'witness'
            or score_plan.get('primary') != 'witness_score'):
        raise ValueError('Score plan does not bind this native witness collection')
    groups = plan['groups']
    if (len(prepared) != len(groups) or type(group) is not int
            or not 0 <= group < len(groups) or groups[group]['role'] != 'test'):
        raise ValueError('Supply all prepared groups in order and select a TEST group')
    samples, origins = [], []
    for index, (root, declared) in enumerate(zip(prepared, groups, strict=True)):
        if declared['role'] not in {'fit', 'test'}:
            raise ValueError('Unknown native prepared-group role')
        bind(root / 'manifest.json', declared['manifest_sha256'])
        manifest = data.read_json(root / 'manifest.json')
        bind(root / 'inputs.jsonl', declared['input_sha256'])
        if manifest.get('inputs_sha256') != declared['input_sha256']:
            raise ValueError('Prepared input binding differs from native plan')
        rows = list(read_samples(root / 'inputs.jsonl'))
        if not rows or len(rows) != manifest['samples']:
            raise ValueError('Prepared inference population count changed')
        if declared['role'] == 'test' and (
                set(manifest.get('coverage', {})) != {'complete'}
                or manifest['selection'].get('limit') is not None
                or manifest['selection'].get('ids')):
            raise ValueError('TEST prepared inputs must be complete and unfiltered')
        for sample in rows:
            _ = sample.inputs
            if not isinstance(sample.source_id, str) or not sample.source_id:
                raise ValueError('Nonempty source identity required')
            samples.append((index, sample))
        origins.append({'group': index, 'original_path': declared['path'],
                        'relocated_path': str(root), 'role': declared['role'],
                        'raw_files_recorded_not_verified': manifest.get('raw_files', {})})
    if len(samples) != len(native_records) or len({s.id for _, s in samples}) != len(samples):
        raise ValueError('Independent native response population changed')
    # Check ALL TEST records before choosing one group; selected score subsets fail.
    expected = []
    for record, (g, sample) in zip(native_records, samples, strict=True):
        if (record['id'] != sample.id or record['group'] != g
                or record['response_sha256'] != data.text_hash(sample.response)):
            raise ValueError('Native record differs from prepared inference identity')
        bind(_member(native, record['file']), record['sha256'])
        if groups[g]['role'] == 'test':
            expected.append((record['id'], g, record['response_sha256']))
    if [(r['id'], r['group'], r['response_sha256']) for r in score_records] != expected:
        raise ValueError('Score records must cover ALL native TEST responses in order')
    score_by_id = {r['id']: r for r in score_records}
    parts, left, right = [], [], []
    for ordinal, (record, (g, sample)) in enumerate(zip(native_records, samples, strict=True)):
        axes, answer, full_offsets = _native_axis(_member(native, record['file']), sample, ordinal)
        if groups[g]['role'] != 'test':
            continue
        scored = score_by_id[sample.id]
        if score_seal[scored['file']] != scored['sha256']:
            raise ValueError('Score record digest differs from score seal')
        with np.load(_member(scores, scored['file']), allow_pickle=False) as saved:
            raw = {k: saved[k] for k in
                   ['target', 'token_id', 'offsets', 'answer_index', 'frozen_score', 'witness_score']}
        # Historical native_dataset.score stores FULL offsets, while target,
        # token_id and answer_index are already on the original valid-token axis.
        if (any(raw[key].dtype.kind not in 'iu'
                for key in ['target', 'token_id', 'offsets', 'answer_index'])
                or not np.array_equal(raw['target'], axes['token_index'])
                or not np.array_equal(raw['token_id'], axes['token_id'])
                or not np.array_equal(raw['offsets'], full_offsets)
                or not np.array_equal(raw['answer_index'], answer)):
            raise ValueError('Score token axes differ from independent native population')
        for key in ['frozen_score', 'witness_score']:
            values = raw[key]
            if (values.shape != axes['token_index'].shape or values.dtype.kind not in 'fiu'
                    or not np.isfinite(values).all()):
                raise ValueError('Finite aligned historical score vectors required')
        if g == group:
            parts.append(axes)
            left.append(raw['frozen_score'])
            right.append(raw['witness_score'])
    population = {key: np.concatenate([part[key] for part in parts]) for key in AXES}
    if not len(population['sample_id']):
        raise ValueError('Selected TEST group has no scored tokens')
    _validate_axes(population, population['token_index'].shape)
    observer = _observer(plan)
    if any(data.file_hash(Path(path)) != digest for path, digest in bindings.items()):
        raise ValueError('Inputs changed during cache export')
    completed_identity(identity)
    output.mkdir(parents=True, exist_ok=False)
    for method, arrays in [('r4', left), ('witness', right)]:
        with (output / (method + '.npz')).open('xb') as stream:
            np.savez_compressed(stream, **population, score=np.concatenate(arrays),
                                method=np.array(method), observer_id=np.array(observer))
    with (output / 'population.npz').open('xb') as stream:
        np.savez_compressed(stream, **population)
    data.write_json(output / 'protocol.json', {**RULE, 'task': taskname, 'observer_id': observer,
        'population_manifest': 'population.npz',
        'population_sha256': data.file_hash(output / 'population.npz')})
    data.write_json(output / 'export.json', {'status': 'exported_not_scored_or_evaluated',
        'task': taskname, 'selected_native_group': group, 'observer_id': observer,
        'responses_scored': len(set(population['sample_id'].tolist())),
        'tokens_scored': len(population['sample_id']), 'all_test_records_verified': len(expected),
        'input_sha256': bindings, 'prepared_origins': origins, 'source': identity,
        'annotation_reads': 0, 'raw_acquisition_files_verified': False,
        'model_weights_reopened': False, 'metrics_computed': False})
    names = ['r4.npz', 'witness.npz', 'population.npz', 'protocol.json', 'export.json']
    data.write_json(output / 'seal.json', {name: data.file_hash(output / name) for name in names})
