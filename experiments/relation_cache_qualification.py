"""Integrated CPU I/O stage for H017's recovered-cache qualification.

The central r4-experiment parser calls ``run(args)``.  This module deliberately
has no standalone launcher, model loader, annotation reader or detector score.
Research-memory check/claim belongs to the caller before invoking this stage.
"""

import hashlib
from pathlib import Path
import time

from .data import file_hash, read_json, write_json
from .provenance import completed_identity, research_identity
from .research.relation_cache import DEFAULT_POLICY, load_capture, qualify_arrays


def _contained(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Qualification input escapes its declared root')
    return path


def capture_identity(capture_root, config_path):
    """Bind the exact observer config and frozen producer source before replay."""
    root = Path(capture_root).resolve()
    environment = read_json(root / 'MODEL_ENVIRONMENT.json')
    expected = environment['files']['config.json']['sha256']
    actual = file_hash(config_path)
    if actual != expected:
        raise ValueError('Observer config differs from captured MODEL_ENVIRONMENT')
    if environment['transformers'] != DEFAULT_POLICY.transformers_version:
        raise ValueError('Producer Transformers version differs from admitted replay')
    code = read_json(root / 'CAPTURE_CODE.json')
    verified = {}
    for original, digest in code.items():
        if 'experiments/' not in original:
            raise ValueError('Unrecognized producer source identity')
        relative = 'experiments/' + original.split('experiments/', 1)[1]
        path = _contained(root / 'capture_code', relative)
        if file_hash(path) != digest:
            raise ValueError('Frozen producer source differs from CAPTURE_CODE')
        verified[relative] = digest
    return {'config_sha256': actual, 'producer_sources_sha256': verified,
            'metadata_sha256': {name: file_hash(root / name) for name in
                               ('MODEL_ENVIRONMENT.json', 'CAPTURE_CODE.json',
                                'CAPTURE_PROTOCOL.json', 'CAPTURE_COMPLETE.json')},
            'captured_runtime': {k: environment[k] for k in
                                 ('python', 'torch', 'transformers', 'numpy')}}


def select_records(protocol, *, ids=None, limit=3):
    """Hash42-ordered sources; finite canaries use one answer per source."""
    candidates = [r for r in protocol['records']
                  if r['task'] == 'QA' and r['split'] == 'test'
                  and r.get('partition') == 'test']
    key = lambda value: hashlib.sha256(f'42:{value}'.encode()).hexdigest()
    candidates.sort(key=lambda r: (key(r['source_id']), key(r['id'])))
    if len({str(r['id']) for r in candidates}) != len(candidates):
        raise ValueError('Duplicate QA TEST answer identity')
    if ids is not None:
        wanted = [str(value) for value in ids]
        if not wanted or len(set(wanted)) != len(wanted):
            raise ValueError('Explicit answer IDs must be nonempty and unique')
        candidates = [r for r in candidates if str(r['id']) in set(wanted)]
        if {str(r['id']) for r in candidates} != set(wanted):
            raise ValueError('Explicit answer ID is outside QA TEST population')
    elif limit is not None:
        if limit < 1:
            raise ValueError('Qualification limit must be positive')
        first = {}
        for row in candidates:
            first.setdefault(str(row['source_id']), row)
        candidates = list(first.values())[:limit]
    if not candidates:
        raise ValueError('No selected QA TEST records')
    return candidates


def run(args):
    """Qualify admitted original caches after the caller's research-memory claim.

    Required namespace attributes: capture_root, config, output.
    Optional: ids=None, limit=3 (None means all), query_chunk=8,
    max_memory_mb=512, prepared_root=None, data_manifest=None.

    ``data_manifest`` is the recovered acquisition manifest, whose relative paths
    are relative to ``data_root``.  Supplying it verifies downloaded payload bytes
    against the preexisting receipt; otherwise this run reports only newly
    witnessed file identities, never a successful receipt comparison.
    """
    started = time.perf_counter()
    root = Path(args.capture_root).resolve()
    output = Path(args.output).resolve()
    config_path = Path(args.config).resolve()
    config = read_json(config_path)
    producer = capture_identity(root, config_path)
    protocol = read_json(root / 'CAPTURE_PROTOCOL.json')
    if (protocol.get('layer') != 15 or protocol.get('local_width') != 8
            or protocol.get('collection_opens_annotation_files') is not False
            or protocol.get('natural_label_fits') != 0):
        raise ValueError('Capture protocol differs from admitted no-label contract')
    selected = select_records(protocol, ids=getattr(args, 'ids', None),
                              limit=getattr(args, 'limit', 3))
    chunk = getattr(args, 'query_chunk', 8)
    budget = int(getattr(args, 'max_memory_mb', 512) * 1024**2)
    if chunk < 1 or budget < 1:
        raise ValueError('Positive query chunk and memory budget required')
    data_root = getattr(args, 'data_root', None)
    manifest_path = getattr(args, 'data_manifest', None)
    receipt = {}
    if manifest_path is not None:
        if data_root is None:
            raise ValueError('data_root is required with an acquisition manifest')
        data_root = Path(data_root).resolve()
        manifest = read_json(manifest_path)
        for row in manifest['files']:
            path = _contained(data_root, row['relative_path'])
            receipt[str(path)] = row['sha256']
    inputs = []
    prepared = getattr(args, 'prepared_root', None)
    for row in selected:
        directory = _contained(root / 'capture', str(row['id']))
        files = {name: directory / name for name in ('record.json', 'arrays.npz')}
        hashes = {name: file_hash(path) for name, path in files.items()}
        if receipt and any(receipt.get(str(path.resolve())) != hashes[name]
                           for name, path in files.items()):
            raise ValueError('Capture payload differs from acquisition receipt')
        witness = {'id': str(row['id']), 'source_id': str(row['source_id']),
                   'files_sha256': hashes}
        if prepared is not None:
            response_path = _contained(Path(prepared).resolve(),
                                       str(row['directory']) + '/response.json')
            witness['prepared_response_sha256'] = file_hash(response_path)
            witness['prepared_response_path'] = str(response_path)
        inputs.append(witness)
    identity = research_identity()
    plan = {'schema': 'h017-cache-qualification-plan-v1',
            'stage': 'engineering_attention_identity_not_detector_evaluation',
            'selection': 'SHA256(42:source_id), then SHA256(42:answer_id); '
                         'finite limit takes first answer per distinct source; QA TEST only',
            'inputs': inputs, 'observer': producer, 'source_runtime': identity,
            'policy': DEFAULT_POLICY.to_dict(), 'query_chunk': chunk,
            'incremental_memory_budget_bytes': budget,
            'acquisition_receipt_verified': bool(receipt),
            'acquisition_manifest_sha256': file_hash(manifest_path) if manifest_path else None,
            'opens_gold_labels': False, 'new_model_forwards': 0}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', plan)  # Frozen before any tensor payload decode.
    results = []
    for row, witness in zip(selected, inputs, strict=True):
        case_started = time.perf_counter()
        directory = _contained(root / 'capture', str(row['id']))
        record = read_json(directory / 'record.json')
        for key in ('id', 'source_id', 'task', 'split', 'directory'):
            if str(record[key]) != str(row[key]):
                raise ValueError('Protocol/record original-response identity differs')
        arrays, admission = load_capture(directory / 'arrays.npz', query_chunk=chunk,
                                         max_memory_bytes=budget)
        response = None
        if prepared is not None:
            response = read_json(_contained(Path(prepared).resolve(),
                                           str(row['directory']) + '/response.json'))
        result = qualify_arrays(arrays, config, record, query_chunk=chunk, response=response)
        result.update(witness, **admission, elapsed_seconds=time.perf_counter() - case_started)
        # Recheck current bytes before publishing a qualified result.
        if any(file_hash(directory / name) != digest
               for name, digest in witness['files_sha256'].items()):
            raise RuntimeError('Capture bytes changed during qualification')
        if prepared is not None and file_hash(witness['prepared_response_path']) != witness['prepared_response_sha256']:
            raise RuntimeError('Prepared response changed during qualification')
        write_json(output / f"answer-{row['id']}.json", result)
        results.append({'id': str(row['id']), 'source_id': str(row['source_id']),
                        'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'],
                        'estimated_incremental_bytes': admission['estimated_incremental_bytes']})
        del arrays
    completed_identity(identity)
    if capture_identity(root, config_path) != producer:
        raise RuntimeError('Producer/config identity changed during qualification')
    counts = {status: sum(r['status'] == status for r in results)
              for status in sorted({r['status'] for r in results})}
    summary = {'schema': 'h017-cache-qualification-result-v1', 'cases': results,
               'status_counts': counts, 'all_qualified': all(r['status'] == 'qualified' for r in results),
               'elapsed_seconds': time.perf_counter() - started, 'model_forwards': 0,
               'labels_read': False, 'detector_effectiveness_evaluated': False}
    write_json(output / 'summary.json', summary)
    write_json(output / 'seal.json', {path.name: file_hash(path)
                                     for path in sorted(output.glob('*.json'))})
    return summary
