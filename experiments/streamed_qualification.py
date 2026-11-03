"""Two-world original-prefix resource/coordinate qualification, without a head.

The central experiment CLI calls run after a research-memory claim.  This stage
never selects a detector, opens annotations, changes tolerances or retries.
"""

import ast
from contextlib import contextmanager, nullcontext
from pathlib import Path
import time

import numpy as np

from .data import file_hash, read_json, write_json
from .provenance import completed_identity, research_identity
from .relation_cache_qualification import capture_identity
from .research.relation_cache import archive_headers
from .research.streamed_observer import ResourceLimits, SafeTensorStore, StreamedObserver, _available_ram


COORDINATE_POLICY = {'fp16_relative': 2.**-11 / (1 - 2.**-11),
                     'fp16_subnormal_absolute': 2.**-25,
                     'fp32_backend_absolute': .002, 'fp32_backend_relative': .0001}
PRODUCER = {
    'experiments/flow_latent/local_graph_transport/capture.py':
        ('002eeb0480280534671396c32d91dc42372884915ebf505e8548a71a9736c75a',
         ('LayerCaptured', 'capture_selected_layer')),
    'experiments/flow_latent/provenance_joint_state/measure.py':
        ('df32ae3415496f3c95a525cbb499d6822cd3beac3a1106a83de9dfa81941bdaf',
         ('block_source_reads',)),
}


def _path(value):
    text = str(value)
    return Path(text[4:] if text.startswith('\\\\?\\') else text).resolve()


def verify_assets(manifest_path, model, *, verification=None, state=None):
    """Require the completed, disarmed and independently verified asset closure."""
    manifest_path, model = _path(manifest_path), _path(model)
    verification = _path(verification or manifest_path.with_name('independent-verification-v1.json'))
    state = _path(state or manifest_path.with_name('mirror-state-v1.json'))
    manifest, receipt, transfer = map(read_json, (manifest_path, verification, state))
    digest = file_hash(manifest_path)
    if (transfer.get('status') != 'complete' or transfer.get('armed') is not False
            or transfer.get('active') != 0 or transfer.get('failed') != 0
            or transfer.get('manifest_sha256') != digest):
        raise ValueError('Model transfer is not successfully complete/disarmed')
    if (receipt.get('status') != 'all9_assets_SHA_verified_291_BF16_headers_index_matched'
            or receipt.get('schema') != 'h017-original-hf-assets-independent-verification-v1'
            or receipt.get('manifest_sha256') != digest
            or receipt.get('weights_match_original_capture_MODEL_ENVIRONMENT') is not True
            or receipt.get('mirror_successfully_disarmed') is not True
            or receipt.get('root_lock_absent') is not True or receipt.get('tensor_count') != 291):
        raise ValueError('Original model assets lack the completed independent verification')
    data_root = _path(manifest['data_root'])
    if data_root.with_name(data_root.name + '.http-mirror.lock').exists():
        raise ValueError('Data-root transfer lock is still active')
    files = manifest['files']
    if len(files) != 9:
        raise ValueError('Original model asset population must contain nine files')
    verified = {r['name']: (r['sha256'], r['bytes']) for r in receipt['files']}
    expected = {}
    for row in files:
        path = (data_root / row['relative_path']).resolve()
        if not path.is_relative_to(data_root) or path.parent != model:
            raise ValueError('Manifest asset escapes the exact model directory')
        if verified.get(path.name) != (row['sha256'], row['size']):
            raise ValueError('Independent asset receipt differs from manifest')
        if path.stat().st_size != row['size']:
            raise ValueError('Verified asset size changed')
        if not path.name.endswith('.safetensors') and file_hash(path) != row['sha256']:
            raise ValueError('Verified model metadata bytes changed')
        expected[path.name] = {'sha256': row['sha256'], 'bytes': row['size']}
    if {p.name for p in model.iterdir()} != set(expected):
        raise ValueError('Unexpected/missing model directory members')
    return {'files': expected, 'manifest_sha256': digest,
            'verification_sha256': file_hash(verification), 'state_sha256': file_hash(state),
            'index_historical_identity': False}


def frozen_capture_functions(capture_root):
    """Extract only explicitly pinned producer functions, with explicit globals.

    These exact, already-reviewed F041 source identities are the allowlist.  No
    import or arbitrary path from the recovered research tree is executed.
    """
    import torch
    namespace = {'torch': torch, 'contextmanager': contextmanager}
    root = Path(capture_root) / 'capture_code'
    for relative, (expected, names) in PRODUCER.items():
        path = root / relative
        if file_hash(path) != expected:
            raise ValueError('Pinned producer extraction source changed')
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        nodes = [node for node in tree.body
                 if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
        if {node.name for node in nodes} != set(names):
            raise ValueError('Pinned producer function population changed')
        selected = ast.Module(body=nodes, type_ignores=[])
        exec(compile(selected, str(path), 'exec'), namespace)
    return namespace


def coordinate_summary(actual, saved, *, head_axis=None):
    """Every coordinate passes the policy frozen before original model calls."""
    actual, saved = np.asarray(actual), np.asarray(saved)
    if actual.dtype != np.float32 or saved.dtype != np.float16 or actual.shape != saved.shape:
        raise ValueError('Coordinate shape or FP32-observed/FP16-archived dtype changed')
    if not np.isfinite(actual).all() or not np.isfinite(saved).all():
        raise ValueError('Nonfinite original-prefix coordinate')
    tolerance = (np.abs(saved.astype(np.float64)) *
                 (COORDINATE_POLICY['fp16_relative'] + COORDINATE_POLICY['fp32_backend_relative'])
                 + COORDINATE_POLICY['fp16_subnormal_absolute']
                 + COORDINATE_POLICY['fp32_backend_absolute'])
    error = np.abs(actual.astype(np.float64) - saved.astype(np.float64))
    result = {'elements': actual.size, 'violations': int((error > tolerance).sum()),
              'max_absolute_error': float(error.max(initial=0)),
              'max_tolerance': float(tolerance.max(initial=0)),
              'max_error_over_tolerance': float((error / tolerance).max(initial=0)),
              'observed_dtype': str(actual.dtype), 'archival_dtype': str(saved.dtype)}
    if head_axis is not None:
        result['physical_heads'] = [coordinate_summary(np.take(actual, i, axis=head_axis),
            np.take(saved, i, axis=head_axis)) for i in range(actual.shape[head_axis])]
    return result


def _resources(device):
    import torch
    result = {'available_physical_ram_bytes': _available_ram()}
    import os
    if os.name == 'nt':
        import ctypes
        class Counters(ctypes.Structure):
            _fields_ = [('size', ctypes.c_ulong), ('faults', ctypes.c_ulong),
                        *[(name, ctypes.c_size_t) for name in
                          ('peak_working', 'working', 'peak_paged', 'paged',
                           'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile', 'private')]]
        counter = Counters()
        counter.size = ctypes.sizeof(counter)
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = ctypes.c_void_p
        memory = ctypes.windll.psapi.GetProcessMemoryInfo
        memory.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        if not memory(process(), ctypes.byref(counter), counter.size):
            raise OSError('Process memory accounting failed')
        result.update(process_working_bytes=counter.working,
                      process_peak_working_bytes=counter.peak_working,
                      process_private_bytes=counter.private,
                      process_peak_commit_bytes=counter.peak_pagefile)
    else:
        import resource
        result['process_peak_working_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if device.type == 'cuda':
        free, total = torch.cuda.mem_get_info(device)
        result.update(gpu_free_bytes=free, gpu_total_bytes=total,
                      gpu_allocated_bytes=torch.cuda.memory_allocated(device),
                      gpu_reserved_bytes=torch.cuda.memory_reserved(device),
                      gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result


def _hook_counts(model):
    return {name: (len(module._forward_pre_hooks), len(module._forward_hooks))
            for name, module in model.named_modules()}


@contextmanager
def _native_dtype_guards(model):
    """Observe native values before the historical capture hooks call .float()."""
    import torch
    observed, handles = {}, []
    layer = model.model.layers[15]
    def check(name, value):
        observed[name] = str(value.dtype)
        if value.dtype != torch.float32:
            raise RuntimeError(f'Native {name} is not FP32 before capture conversion')
    handles.append(layer.register_forward_pre_hook(lambda module, args: check('residual_input', args[0])))
    for name, module in [('query', layer.self_attn.q_proj), ('key', layer.self_attn.k_proj),
                         ('value', layer.self_attn.v_proj), ('attention_write', layer.self_attn.o_proj),
                         ('mlp_write', layer.mlp)]:
        def observe(module, args, output, name=name):
            check(name, output)
        handles.append(module.register_forward_hook(observe))
    try:
        yield observed
    finally:
        for handle in handles:
            handle.remove()


def _reference_payload(path, p, t, n):
    with np.load(path, allow_pickle=False) as archive:
        names = ('token_ids', 'answer_ids', 'source_mask', 'query_positions', 'nodes', 'query', 'key', 'value')
        reference = {name: archive[name] for name in names}
    for name, length in [('token_ids', n), ('answer_ids', t), ('query_positions', t+1)]:
        if reference[name].dtype != np.int64 or reference[name].shape != (length,):
            raise ValueError('Archived integer token/query axis dtype or shape changed')
    source = reference['source_mask']
    if source.dtype != np.bool_ or source.shape != (n,):
        raise ValueError('Archived source mask is not the exact boolean 1-D axis')
    if (not np.array_equal(reference['token_ids'][p:], reference['answer_ids'])
            or not np.array_equal(reference['query_positions'], np.arange(p-1, n))
            or source[p:].any() or (reference['token_ids'] < 0).any()
            or (reference['token_ids'] >= 128256).any()):
        raise ValueError('Archived original token/query/source axes changed')
    return reference


def _validate_reference(headers, record, maximum):
    audit = record['audit']
    p, t, n = audit['prompt_length'], audit['answer_tokens'], audit['input_tokens']
    if p > maximum or maximum > 256 or p + t != n or audit['layer'] != 15:
        raise ValueError('Original full prompt exceeds the admitted prefix contract')
    if (record['task'] != 'QA' or record['split'] != 'test' or audit['stop_after_layer'] is not True
            or audit['full_model_forwards'] != 0
            or [w['blocked'] for w in audit['worlds']] != [False, True]
            or any(w['executed_layers'] != 16 or w['full_model_completed'] is not False
                   for w in audit['worlds'])):
        raise ValueError('Archived world/precision/timing contract changed')
    shapes = {'nodes': (2, t+1, 3, 4096), 'query': (2, t+1, 32, 128),
              'key': (2, n, 8, 128), 'value': (2, n, 8, 128)}
    if tuple(audit['sites']) != ('residual', 'attention_write', 'mlp_write'):
        raise ValueError('Archived node sites changed')
    for name, shape in shapes.items():
        if tuple(headers[name]['shape']) != shape or headers[name]['dtype'] != 'float16':
            raise ValueError('Archived coordinate header differs from qualification contract')
    return p, t, n


def run(args):
    """Root-claimed original-prefix canary; exactly two early-stop model calls.

    Namespace: capture_root, model, asset_manifest, output; optional
    asset_verification=None, asset_state=None, device='cuda:0', max_tokens=256,
    id='16465'.  There is no automatic retry or precision/input substitution.
    """
    import torch
    started = time.perf_counter()
    capture, model, output = _path(args.capture_root), _path(args.model), _path(args.output)
    sample = str(getattr(args, 'id', '16465'))
    if not sample.isdecimal():
        raise ValueError('Capture ID must be a decimal original answer identity')
    assets = verify_assets(args.asset_manifest, model,
        verification=getattr(args, 'asset_verification', None), state=getattr(args, 'asset_state', None))
    observer_identity = capture_identity(capture, model / 'config.json')
    if any(assets['files'].get(name) != row for name, row in
           read_json(capture / 'MODEL_ENVIRONMENT.json')['files'].items()):
        raise ValueError('Model assets differ from the capture-attested file identities')
    directory = capture / 'capture' / sample
    record = read_json(directory / 'record.json')
    if str(record['id']) != sample:
        raise ValueError('Capture answer identity differs')
    maximum = getattr(args, 'max_tokens', 256)
    p, t, n = _validate_reference(archive_headers(directory / 'arrays.npz'), record, maximum)
    source_identity = research_identity()
    limits = ResourceLimits(max_sequence_tokens=maximum)
    device = torch.device(getattr(args, 'device', 'cuda:0'))
    plan = {'schema': 'h017-original-prefix-qualification-plan-v1', 'id': sample,
            'source_id': str(record['source_id']), 'original_input_tokens': n,
            'complete_original_prompt_tokens': p, 'answer_tokens': t,
            'query_position': p-1, 'reference_row': 0, 'executed_layers': 16,
            'planned_model_calls': 2, 'full_model_completed': False,
            'source_blocked_layers': list(range(16)), 'coordinate_policy': dict(COORDINATE_POLICY),
            'policy_scope': 'FP16 rounding plus declared FP32 backend budget; not a network theorem',
            'comparisons': ['nodes3sites', 'raw_preRoPE_Q32', 'raw_preRoPE_K8_prefix', 'V8_prefix'],
            'assets': assets, 'observer_capture_identity': observer_identity,
            'source_runtime': source_identity, 'resource_limits': limits.__dict__,
            'before': _resources(device),
            'capture_sha256': {name: file_hash(directory / name) for name in ('record.json', 'arrays.npz')},
            'precision': {'weight_storage': 'bfloat16', 'linear_compute': 'float32',
                          'embedding_output': 'float32', 'native_hidden_compute': 'float32',
                          'attention_backend': 'sdpa', 'autocast': False, 'tf32': False},
            'labels_read': False, 'final_norm_or_head_calls': 0}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', plan)
    worlds, actual_calls = [], 0
    observer = None
    try:
        functions = frozen_capture_functions(capture)
        reference = _reference_payload(directory / 'arrays.npz', p, t, n)
        config = read_json(model / 'config.json')
        store = SafeTensorStore(model, index_sha256=assets['files']['model.safetensors.index.json']['sha256'],
                                expected_files=assets['files'])
        observer = StreamedObserver(store, config,
            config_sha256=assets['files']['config.json']['sha256'], device=device, limits=limits)
        base_hooks = _hook_counts(observer.native)
        for world, blocked in enumerate((False, True)):
            world_started = time.perf_counter()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            before = _resources(device)
            block = (functions['block_source_reads'](observer.native,
                     reference['source_mask'][:p].tolist(), p) if blocked else nullcontext())
            with torch.inference_mode(), block, _native_dtype_guards(observer.native) as native_dtypes:
                precision = {'inference_mode': torch.is_inference_mode_enabled(),
                             'grad_enabled': torch.is_grad_enabled(),
                             'autocast_cpu': torch.is_autocast_enabled('cpu'),
                             'autocast_cuda': torch.is_autocast_enabled('cuda'),
                             'cuda_matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                             'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                             'rotary_inverse_frequency_dtype': str(observer.native.model.rotary_emb.inv_freq.dtype),
                             'weight_parameter_dtypes': sorted({str(v.dtype) for v in observer.native.parameters()})}
                if (precision['grad_enabled'] or precision['autocast_cpu'] or precision['autocast_cuda']
                        or precision['cuda_matmul_tf32'] or precision['cudnn_tf32']
                        or precision['weight_parameter_dtypes'] != ['torch.bfloat16']
                        or precision['rotary_inverse_frequency_dtype'] != 'torch.float32'):
                    raise RuntimeError('Original-prefix execution precision flags changed')
                with functions['capture_selected_layer'](observer.native, 15, p-1,
                                                          stop_after_layer=True) as captured:
                    try:
                        actual_calls += 1
                        observer.native.model(observer.input_ids(reference['token_ids'][:p].tolist()),
                                              use_cache=False)
                    except functions['LayerCaptured']:
                        pass
                    else:
                        raise RuntimeError('Selected-layer early exit did not occur')
            observed = {'nodes': np.stack([captured[name][0].numpy() for name in
                        ('residual', 'attention_write', 'mlp_write')]),
                        'query': captured['query'][0].numpy().reshape(32, 128),
                        'key': captured['key'].numpy().reshape(p, 8, 128),
                        'value': captured['value'].numpy().reshape(p, 8, 128)}
            expected = {'nodes': reference['nodes'][world, 0], 'query': reference['query'][world, 0],
                        'key': reference['key'][world, :p], 'value': reference['value'][world, :p]}
            metrics = {name: coordinate_summary(value, expected[name],
                       head_axis=None if name == 'nodes' else -2) for name, value in observed.items()}
            clean = _hook_counts(observer.native) == base_hooks
            report = observer.resource_report()
            passed = (all(m['violations'] == 0 for m in metrics.values()) and clean
                      and report['all_parameters_meta'] and report['active_module'] is None)
            result = {'world': 'blocked' if blocked else 'native', 'qualified': passed,
                      'metrics': metrics, 'hook_cleanup': clean, 'resource_report': report,
                      'before': before, 'after': _resources(device),
                      'precision_flags': precision,
                      'native_dtypes_before_capture_cast': native_dtypes,
                      'elapsed_seconds': time.perf_counter()-world_started,
                      'full_model_completed': False, 'executed_layers': 16,
                      'original_generator_internal_state': False}
            np.savez_compressed(output / f"{result['world']}-coordinates.npz", **observed)
            write_json(output / f"{result['world']}.json", result)
            worlds.append(result)
            del captured, observed
    except BaseException as error:
        write_json(output / 'engineering-failure.json', {'type': type(error).__name__,
            'message': str(error), 'model_calls_started': actual_calls,
            'worlds_completed': len(worlds), 'resources': _resources(device),
            'elapsed_seconds': time.perf_counter()-started, 'automatic_retry': False})
        raise
    finally:
        if observer is not None:
            observer.close()
    completed_identity(source_identity)
    if verify_assets(args.asset_manifest, model, verification=getattr(args, 'asset_verification', None),
                     state=getattr(args, 'asset_state', None)) != assets:
        raise RuntimeError('Asset gate identities changed during qualification')
    if any(file_hash(directory / name) != digest for name, digest in plan['capture_sha256'].items()):
        raise RuntimeError('Archived reference bytes changed during qualification')
    summary = {'schema': 'h017-original-prefix-qualification-result-v1',
               'all_qualified': len(worlds) == 2 and all(w['qualified'] for w in worlds),
               'model_calls': actual_calls, 'full_model_completed': False,
               'final_norm_or_head_calls': 0, 'labels_read': False,
               'detector_effectiveness_evaluated': False,
               'elapsed_seconds': time.perf_counter()-started,
               'worlds': [{'world': w['world'], 'qualified': w['qualified']} for w in worlds]}
    write_json(output / 'summary.json', summary)
    write_json(output / 'seal.json', {path.name: file_hash(path)
                                     for path in sorted(output.iterdir()) if path.is_file()})
    return summary
