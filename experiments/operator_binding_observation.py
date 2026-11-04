"""Finite original-response source→Norm+MLP readout, with explicit resource gates.

The first gate is a two-response engineering/counterexample assay. It never opens
annotations or establishes detector efficacy, and never resumes a failed attempt.
"""

from pathlib import Path
import time
import zipfile

import numpy as np

from myproject.measure import answer_axis, token_scores
from .data import file_hash, read_json, text_hash, verify_seal, write_json
from .datasets import Sample, read_jsonl
from .provenance import completed_identity, research_identity
from .relation_cache_qualification import capture_identity
from .relation_intervention import _seal, head_prerequisite
from .research.streamed_observer import ResourceLimits, SafeTensorStore, StreamedObserver
from .research.streamed_observer import _available_ram
from .streamed_head_qualification import allocator_cleanup
from .streamed_qualification import _resources, verify_assets


CASE_ORDER = ('5682', '5684')
BRANCH_ORDER = ('native', 'source', 'sham', 'direct_source_write')
MAX_STAGING_BYTES = 8 * 1024**2


def stream_arrays(path, arrays, *, counts, max_chunk_elements=None):
    """One immutable ZIP of NPY members, at most 8 MiB CPU staging per copy.

    This chunks storage only. It never chunks a head/MLP/norm computation.
    Arrays are consumed sequentially, without collecting a CPU layer dictionary.
    """
    import torch
    path = Path(path)
    with path.open('xb') as destination, zipfile.ZipFile(
            destination, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for name, value in arrays.items():
            if not name.replace('_', '').isalnum():
                raise ValueError('Numerical member must have a simple unique name')
            if isinstance(value, torch.Tensor):
                value = value.detach().contiguous()
                if value.dtype == torch.bfloat16:
                    # A rejected dtype is still evidence. Preserve its exact
                    # 16-bit payload; never cast it into accepted FP32 outputs.
                    name += '_bf16bits'
                    value = value.view(torch.uint16)
                dtype = value.reshape(-1)[:0].cpu().numpy().dtype
                shape = tuple(value.shape)
                flat = value.reshape(-1)
            else:
                value = np.ascontiguousarray(value)
                dtype, shape, flat = value.dtype, value.shape, value.reshape(-1)
            if dtype.hasobject:
                raise ValueError('Object arrays are not numerical evidence')
            with archive.open(name + '.npy', 'w', force_zip64=True) as member:
                np.lib.format.write_array_header_1_0(member, {
                    'descr': np.lib.format.dtype_to_descr(dtype),
                    'fortran_order': False, 'shape': shape})
                step = max(1, MAX_STAGING_BYTES // dtype.itemsize)
                if max_chunk_elements is not None:
                    step = min(step, max_chunk_elements)
                for first in range(0, flat.numel() if isinstance(flat, torch.Tensor) else flat.size, step):
                    part = flat[first:first + step]
                    if isinstance(part, torch.Tensor):
                        if (part.device.type == 'cuda' and _available_ram()
                                < 512 * 1024**2 + part.numel() * part.element_size()):
                            raise MemoryError('CPU staging would cross the fixed host reserve')
                        host = part.cpu().numpy()
                        counts['host_transfer_bytes'] = counts.get('host_transfer_bytes', 0) + host.nbytes
                    else:
                        host = part
                    member.write(memoryview(host).cast('B'))
                    counts['max_host_staging_bytes'] = max(
                        counts.get('max_host_staging_bytes', 0), host.nbytes)
                    del host, part


def load_cases(axes_root, prepared, frontend):
    """Read only inference fields and the already qualified original axes."""
    axes_root, prepared = Path(axes_root), Path(prepared)
    verify_seal(axes_root, required=('summary.json', 'plan.json',
        'axes-5682.npz', 'axes-5684.npz', 'case-5682.json', 'case-5684.json'))
    summary, contract = read_json(axes_root/'summary.json'), read_json(frontend)
    if (summary.get('completed') is not True or summary.get('model_forwards') != 0
            or summary.get('annotation_reads') != 0
            or tuple(row['id'] for row in summary['cases']) != CASE_ORDER):
        raise ValueError('The finite tokenizer-only prerequisite changed')
    input_path = prepared/'inputs.jsonl'
    if file_hash(input_path) != contract['evidence_bindings']['input_sha256']:
        raise ValueError('Original inference input SHA changed')
    samples = {str(row['id']): Sample(**row) for row in read_jsonl(input_path)
               if str(row['id']) in CASE_ORDER}
    if set(samples) != set(CASE_ORDER):
        raise ValueError('Two fixed original responses required')
    cases = {}
    for expected, row in zip(contract['cases'], summary['cases'], strict=True):
        key = row['id']
        sample, metadata = samples[key], read_json(axes_root/f'case-{key}.json')
        response_path = Path(expected['primitive_response_path'])
        if (expected['id'] != key or expected['source_id'] != sample.source_id
                or text_hash(sample.response) != expected['response_sha256']
                or file_hash(response_path) != expected['primitive_response_sha256']):
            raise ValueError('Original response/source provenance changed')
        response = read_json(response_path)
        with np.load(axes_root/f'axes-{key}.npz', allow_pickle=False) as saved:
            axes = {name: saved[name] for name in saved.files}
        p, t, n = row['P'], row['T'], row['N']
        if (p != 803 or (t, n) != {'5682': (174, 977), '5684': (233, 1036)}[key]
                or axes['token_ids'].dtype != np.int64
                or axes['source_mask'].dtype != np.bool_
                or axes['token_ids'].shape != (n,)
                or axes['source_mask'].shape != (n,)
                or axes['answer_ids'].tolist() != response['answer_ids']
                or axes['token_ids'][p:].tolist() != response['answer_ids']
                or axes['prechoice_positions'].tolist() != list(range(p - 1, p + t - 1))
                or axes['answer_offsets'].tolist() != response['offsets']
                or axes['source_mask'][p:].any()
                or int(axes['source_mask'].sum()) != 707
                or response['text'] != sample.response):
            raise ValueError('Complete original prechoice/source/response axis changed')
        axis = answer_axis(response, metadata['native_collection_answer_index'])
        if any(not np.array_equal(axis[name], axes[name]) for name in axis):
            raise ValueError('Original mainline packed axis changed')
        cases[key] = dict(sample=sample, response=response, axis=axis,
            metadata=metadata, axes=axes, P=p, T=t, N=n)
    return cases


def prepare(args):
    """Freeze a metadata-only execution plan; no weight numeric payload/model."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    identity = research_identity()
    protocol = read_json(args.protocol)
    if (protocol.get('schema') != 'h018-local-norm-mlp-phase1-v1'
            or protocol.get('case_order') != list(CASE_ORDER)
            or protocol.get('budget') != {'model_calls': 2, 'head_calls': 8,
                'extra_mlp_calls': 256, 'extra_norm_calls': 256,
                'final_norm_replays': 8, 'partition_o_calls': 192, 'av_replays': 64}):
        raise ValueError('Only the independently admitted finite contract is executable')
    for path, expected in protocol['scientific_bindings'].items():
        if file_hash(Path(path)) != expected:
            raise ValueError('Frozen scientific admission/review/resource binding changed')
    cases = load_cases(args.axes, args.prepared, args.frontend_contract)
    assets = verify_assets(args.asset_manifest, args.model,
        verification=args.asset_verification, state=args.asset_state)
    observer_identity = capture_identity(Path(args.capture_root), Path(args.model)/'config.json')
    head = head_prerequisite(Path(args.head_qualification), assets, observer_identity)
    config = read_json(Path(args.model)/'config.json')
    if (config['num_hidden_layers'], config['hidden_size'], config['vocab_size']) != (32, 4096, 128256):
        raise ValueError('Only the original fixed observer configuration is admitted')
    plan = dict(schema='h018-local-norm-mlp-execution-plan-v1',
        protocol=protocol, source_runtime=completed_identity(identity), assets=assets,
        observer_capture_identity=observer_identity, head_prerequisite=head,
        frontend_sha256=file_hash(args.frontend_contract),
        axes_seal_sha256=file_hash(Path(args.axes)/'seal.json'),
        inputs_sha256=file_hash(Path(args.prepared)/'inputs.jsonl'),
        cases={key: {name: row[name] for name in ('P', 'T', 'N')} for key, row in cases.items()},
        model_calls=0, annotation_reads=0, original_generator_causality=False,
        historical_prompt_attestation=False, stage='prepare_only',
        args={name: str(getattr(args, name)) for name in ('protocol', 'axes', 'prepared',
            'frontend_contract', 'model', 'asset_manifest', 'asset_verification',
            'asset_state', 'head_qualification', 'capture_root', 'device')})
    write_json(output/'plan.json', plan)
    _seal(output)
    return plan


def _flat_tensors(value, prefix=''):
    """Flatten nested raw capture dictionaries without copying their tensors."""
    import torch
    arrays = {}
    for name, item in value.items():
        key = prefix + name
        if isinstance(item, (torch.Tensor, np.ndarray)):
            arrays[key] = item
        elif isinstance(item, dict):
            arrays.update(_flat_tensors(item, key + '_'))
    return arrays


def _norm_replay(observer, raw, rows, removal, *, name, output, counts):
    """Same complete N shape/layout; only original prechoice rows are changed."""
    import torch
    candidate = raw.clone(memory_format=torch.preserve_format)
    if removal is not None:
        candidate[:, rows, :] = raw[:, rows, :] - removal
    if candidate.shape != raw.shape or candidate.stride() != raw.stride():
        raise RuntimeError('Full-N finalnorm layout changed')
    counts['final_norm_replays'] = counts.get('final_norm_replays', 0) + 1
    counts['final_norm_rows'] = counts.get('final_norm_rows', 0) + raw.shape[1]
    normalized = observer.native.model.norm(candidate)
    stream_arrays(output/f'finalnorm-{name}.npz', {
        'input': candidate, 'output': normalized,
        'input_rms': (candidate.square().mean(-1) + observer.native.model.norm.variance_epsilon).sqrt(),
    }, counts=counts)
    if (normalized.dtype != torch.float32 or normalized.shape != raw.shape
            or not torch.isfinite(normalized).all().item()):
        raise RuntimeError('Finalnorm replay is not finite full-N FP32')
    selected = normalized[:, rows, :].clone()
    return normalized if removal is None else selected


def _head(observer, hidden, target_ids, output, name, counts):
    import torch
    from .research.operator_binding import full_vocab_endpoints
    t = len(target_ids)
    counts['head_calls'] = counts.get('head_calls', 0) + 1
    counts['head_rows'] = counts.get('head_rows', 0) + t
    start = time.monotonic()
    logits = observer.logits(hidden)
    stream_arrays(output/f'logits-{name}.npz', {'logits': logits[0]}, counts=counts,
                  max_chunk_elements=16*observer.native.config.vocab_size)
    if (logits.dtype != torch.float32
            or tuple(logits.shape) != (1, t, observer.native.config.vocab_size)):
        raise RuntimeError('Original full-vocabulary head axis changed')
    endpoints = full_vocab_endpoints(logits[0], target_ids)
    stream_arrays(output/f'endpoints-{name}.npz', endpoints, counts=counts)
    counts['head_elapsed_seconds'] = counts.get('head_elapsed_seconds', 0.) + time.monotonic() - start
    del logits
    allocator_cleanup(observer)
    return endpoints


def observe_case(observer, case, output, counts):
    """One unchanged complete native forward, four full heads, no retries."""
    import torch
    from myproject.model import attention_backend
    from .pipeline import _validate_prediction
    from .research.operator_binding import _same_bits, score_endpoints
    from .research.operator_capture import capture_local_effects

    output = Path(output)
    output.mkdir(exist_ok=False)
    raw = output/'raw'
    raw.mkdir()
    axes = case['axes']
    ids, positions = axes['token_ids'].tolist(), axes['prechoice_positions'].tolist()
    rows = torch.as_tensor(positions, dtype=torch.long, device=observer.device)
    targets = torch.as_tensor(axes['answer_ids'], dtype=torch.long, device=observer.device)
    write_json(output/'axis.json', dict(id=case['sample'].id, P=case['P'], T=case['T'], N=case['N'],
        original_response_sha256=text_hash(case['sample'].response),
        prechoice_positions=positions, source_mask_tokens=int(axes['source_mask'].sum()),
        historical_prompt_attested=False, annotation_reads=0))
    stream_arrays(raw/'axes.npz', axes, counts=counts)
    before = dict(counts)
    native, captured, normalized, result = None, None, None, None
    start = time.monotonic()

    def on_layer(index, arrays):
        stream_arrays(raw/f'layer-{index:02d}.npz', _flat_tensors(arrays), counts=counts)
        if isinstance(arrays.get('gates'), dict):
            write_json(raw/f'layer-{index:02d}-gates.json', arrays['gates'])

    try:
        with torch.inference_mode(), attention_backend(observer.native, 'eager'):
            with capture_local_effects(observer, ids, axes['source_mask'].tolist(), positions,
                                       counts=counts, on_layer=on_layer) as captured:
                counts['model_calls'] = counts.get('model_calls', 0) + 1
                write_json(output/'model-attempt.json', dict(actual_counts=dict(counts),
                    complete_sequence_tokens=len(ids), automatic_retry=False, backend='eager'))
                native = observer.hidden(ids, use_cache=True)
        # The capture owns only selected rows plus raw final residuals. Save native
        # complete cache layerwise and release it before the large vocabulary head.
        stream_arrays(raw/'native-final.npz', dict(
            postnorm=native.last_hidden_state,
            **({'raw': captured['H_raw']} if captured.get('H_raw') is not None else {})), counts=counts)
        if (native.last_hidden_state.dtype != torch.float32
                or tuple(native.last_hidden_state.shape) != (1, case['N'], observer.native.config.hidden_size)):
            raise RuntimeError('Complete native final hidden contract changed')
        for index, pair in enumerate(native.past_key_values.to_legacy_cache()):
            stream_arrays(raw/f'kv-{index:02d}.npz', dict(key=pair[0], value=pair[1]), counts=counts)
        del pair
        # Stable capture interface is independently tested against the same native
        # postnorm rows; never substitute last_hidden_state for raw H.
        raw_h = captured['H_raw']
        actual_postnorm = native.last_hidden_state
        if not _same_bits(captured['native_postnorm'], actual_postnorm):
            raise RuntimeError('Captured actual finalnorm differs from native output')
        aggregate = captured['accumulation']
        stream_arrays(raw/'aggregate.npz', _flat_tensors(aggregate), counts=counts)
        write_json(raw/'capture-gates.json', captured['gates'])
        write_json(raw/'layer-gates.json', {'layers': captured['layer_gates']})
        if not captured['gates']['passed'] or not aggregate['gates']['passed']:
            raise RuntimeError('Native capture/ordered accumulation numerical gate failed')
        selected_native = actual_postnorm[:, rows, :].clone()
        native = None
        captured['native_postnorm'] = None
        allocator_cleanup(observer)
        with torch.inference_mode():
            normalized = _norm_replay(observer, raw_h, rows, None,
                name='m0', output=raw, counts=counts)
            if not _same_bits(normalized[:, rows, :], selected_native):
                raise RuntimeError('Complete-N no-op finalnorm failed bitwise original rows')
            if not _same_bits(normalized, actual_postnorm):
                raise RuntimeError('Complete-N no-op finalnorm failed bitwise')
            del normalized, actual_postnorm
            endpoints = {'native': _head(observer, selected_native, targets, raw, 'native', counts)}
            del selected_native
            for name, delta in (('source', aggregate['source32']),
                                ('sham', aggregate['sham32']),
                                ('direct_source_write', aggregate['p006_source32'])):
                selected = _norm_replay(observer, raw_h, rows, delta,
                    name=name, output=raw, counts=counts)
                endpoints[name] = _head(observer, selected, targets, raw, name, counts)
                del selected
            readouts = {name: score_endpoints(endpoints['native'], endpoints[name])
                        for name in BRANCH_ORDER[1:]}
            stream_arrays(output/'readouts.npz', {
                **{f'{name}_{field}': value for name, record in readouts.items()
                   for field, value in record.items() if isinstance(value, torch.Tensor)},
                'native_nll': -endpoints['native']['logp'],
            }, counts=counts)
            if any(not value['gates']['passed'] for value in readouts.values()):
                raise RuntimeError('Frozen full-vocabulary decomposition gate failed')
            primary = readouts['source']['risk'].cpu().numpy()[case['axis']['target']]
        prediction = dict(id=case['sample'].id, source_id=case['sample'].source_id,
            response_sha256=text_hash(case['sample'].response),
            method='h018_source_norm_mlp_local_direct_effect_phase1',
            tokens=token_scores(case['axis'], case['response'], primary),
            first_gate_only=True, unknown_semantics_retained=True)
        _validate_prediction(prediction, case['sample'])
        write_json(output/'prediction.json', prediction)
        result = dict(completed=True, actual_counts=dict(counts), counts_before=before,
            original_targets=len(prediction['tokens']), elapsed_seconds=time.monotonic()-start,
            gates=captured['gates'], aggregate_gates=aggregate['gates'],
            endpoint_gates={name: value['gates'] for name, value in readouts.items()},
            resources=_resources(observer.device), automatic_retry=False, annotation_reads=0,
            auroc=None, phase2_admitted=False, scientific_sign_validated=False)
        write_json(output/'result.json', result)
        _seal(raw)
        _seal(output)
        return result
    except BaseException as error:
        if captured is not None:
            partial_final = {name: captured[key] for name, key in (
                ('raw', 'H_raw'), ('postnorm', 'native_postnorm'))
                if isinstance(captured.get(key), torch.Tensor)}
            retention_error = None
            if partial_final and not (raw/'native-final.npz').exists():
                try:
                    stream_arrays(raw/'partial-native-final.npz', partial_final, counts=counts)
                except Exception as retention_failure:
                    retention_error = dict(type=type(retention_failure).__name__, message=str(retention_failure))
            write_json(output/'partial-capture.json', {
                **{key: captured.get(key) for key in ('completed_layers', 'layer_gates', 'gates')},
                'partial_final_retention_error': retention_error})
        write_json(output/'failure.json', dict(type=type(error).__name__, message=str(error),
            actual_counts=dict(counts), counts_before=before, automatic_retry=False,
            resources=_resources(observer.device), annotation_reads=0))
        _seal(raw)
        _seal(output)
        raise
    finally:
        native, captured, normalized = None, None, None


def run(args):
    """Unified CLI stage: explicit metadata plan or one bounded claimed execution."""
    import torch
    if args.prepare_only:
        if args.plan is not None:
            raise ValueError('Preparation cannot consume an execution plan')
        return prepare(args)
    if args.plan is None:
        raise ValueError('Execution requires a separately sealed --plan directory')
    plan_root = Path(args.plan)
    verify_seal(plan_root, required=('plan.json',))
    plan = read_json(plan_root/'plan.json')
    identity = research_identity()
    if identity != plan['source_runtime']:
        raise ValueError('Installed source/runtime differs from the frozen metadata plan')
    for name, expected in plan['args'].items():
        if str(getattr(args, name)) != expected:
            raise ValueError(f'Frozen execution argument changed: {name}')
    if file_hash(args.protocol) != file_hash(Path(plan['args']['protocol'])):
        raise ValueError('Protocol identity changed')
    if read_json(args.protocol) != plan['protocol']:
        raise ValueError('Frozen finite protocol changed')
    for path, digest in plan['protocol']['scientific_bindings'].items():
        if file_hash(Path(path)) != digest:
            raise ValueError('Scientific admission/review bytes changed after preparation')
    if file_hash(Path(args.axes)/'seal.json') != plan['axes_seal_sha256']:
        raise ValueError('Original operational axes seal changed')
    if file_hash(args.frontend_contract) != plan['frontend_sha256']:
        raise ValueError('Original source license/temporal contract changed')
    cases = load_cases(args.axes, args.prepared, args.frontend_contract)
    assets = verify_assets(args.asset_manifest, args.model,
        verification=args.asset_verification, state=args.asset_state)
    if assets != plan['assets']:
        raise ValueError('Original observer assets differ from the execution plan')
    observer_identity = capture_identity(Path(args.capture_root), Path(args.model)/'config.json')
    if observer_identity != plan['observer_capture_identity']:
        raise ValueError('Observer capture prerequisite changed after preparation')
    if head_prerequisite(Path(args.head_qualification), assets, observer_identity) != plan['head_prerequisite']:
        raise ValueError('Native head/cache/no-op prerequisite changed after preparation')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    counts = {name: 0 for name in plan['protocol']['budget']}
    counts.update(host_transfer_bytes=0, max_host_staging_bytes=0)
    write_json(output/'plan.json', plan)
    observer, results = None, {}
    start = time.monotonic()
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(args.device)
        resources = _resources(torch.device(args.device))
        resources['torch'] = torch.__version__
        resources['cuda_runtime'] = torch.version.cuda
        resources['device_capability'] = list(torch.cuda.get_device_capability(args.device))
        resources['device_name'] = torch.cuda.get_device_name(args.device)
        write_json(output/'post-import-resource-admission.json', resources)
        if (torch.__version__ != '2.8.0+cu128' or torch.version.cuda != '12.8'
                or resources['device_capability'] != [12, 0]):
            raise RuntimeError('Frozen original SM120/Torch2.8+cu128 execution contract changed')
        if (resources['available_physical_ram_bytes'] < 512*1024**2
                or resources.get('gpu_free_bytes', 0) < 512*1024**2):
            raise MemoryError('Fixed post-import host/GPU reserve not met')
        files = assets['files']
        store = SafeTensorStore(args.model,
            index_sha256=files['model.safetensors.index.json']['sha256'], expected_files=files)
        limits = ResourceLimits(max_sequence_tokens=max(case['N'] for case in cases.values()))
        with StreamedObserver(store, read_json(Path(args.model)/'config.json'),
                config_sha256=files['config.json']['sha256'], limits=limits,
                device=args.device) as observer:
            for key in CASE_ORDER:
                results[key] = observe_case(observer, cases[key], output/f'case-{key}', counts)
                write_json(output/f'progress-{key}.json', dict(completed=list(results), actual_counts=dict(counts)))
            if any(counts[name] != expected for name, expected in plan['protocol']['budget'].items()):
                raise RuntimeError('Completed execution differs from the finite call budget')
            resource_report = observer.resource_report()
        report = dict(completed=True, phase='first_engineering_counterexample_gate_only',
            actual_counts=counts, cases=results, resource_report=resource_report,
            elapsed_seconds=time.monotonic()-start, annotation_reads=0, auroc=None,
            phase2_admitted=False, full900_admitted=False, detector_validated=False,
            source_runtime=completed_identity(identity), automatic_retry=False)
        write_json(output/'result.json', report)
        _seal(output)
        return report
    except BaseException as error:
        write_json(output/'failure.json', dict(type=type(error).__name__, message=str(error),
            actual_counts=counts, completed=list(results), automatic_retry=False,
            elapsed_seconds=time.monotonic()-start, annotation_reads=0,
            resources=_resources(torch.device(args.device))))
        _seal(output)
        raise
