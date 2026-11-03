"""Full native observer head/cache/no-op engineering qualification only."""

from contextlib import contextmanager
import hashlib
from pathlib import Path
import time

import numpy as np

from .data import file_hash, read_json, verify_seal, write_json
from .provenance import completed_identity, research_identity
from .relation_cache_qualification import capture_identity
from .research.relation_cache import archive_headers
from .research.streamed_observer import ResourceLimits, SafeTensorStore, StreamedObserver
from .streamed_qualification import (_hook_counts, _path, _reference_payload,
                                     _resources, _validate_reference, verify_assets)


POLICY = {'hidden_absolute': .002, 'hidden_relative': .0001,
          'logit_absolute': .002, 'logit_relative': .0001,
          'original_token_logp_absolute': .002, 'noop': 'bitwise_equal'}


def compare_outputs(reference, candidate):
    result = {}
    for name in ('hidden', 'logits'):
        expected, actual = reference[name], candidate[name]
        if (expected.dtype != np.float32 or actual.dtype != np.float32
                or expected.shape != actual.shape or not np.isfinite(expected).all()
                or not np.isfinite(actual).all()):
            raise ValueError('Full-output FP32 coordinate contract changed')
        stem = 'logit' if name == 'logits' else 'hidden'
        tolerance = POLICY[f'{stem}_absolute'] + POLICY[f'{stem}_relative'] * np.abs(expected.astype(np.float64))
        error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        result[name] = {'elements': actual.size, 'violations': int((error > tolerance).sum()),
                        'max_absolute_error': float(error.max(initial=0)),
                        'max_error_over_tolerance': float((error/tolerance).max(initial=0)),
                        'bitwise_equal': expected.tobytes() == actual.tobytes()}
    if not np.isfinite([reference['logp'], candidate['logp']]).all():
        raise ValueError('Original-token normalized logp is not finite')
    delta = abs(reference['logp'] - candidate['logp'])
    result['logp'] = {'absolute_error': delta,
                      'within_budget': delta <= POLICY['original_token_logp_absolute'],
                      'bitwise_equal': np.float32(reference['logp']).tobytes() == np.float32(candidate['logp']).tobytes()}
    result['within_budget'] = (all(result[n]['violations'] == 0 for n in ('hidden', 'logits'))
                               and result['logp']['within_budget'])
    result['bitwise_equal'] = all(result[n]['bitwise_equal'] for n in ('hidden', 'logits', 'logp'))
    return result


def _cache_tensors(cache, config, length):
    import torch
    pairs = cache.to_legacy_cache()
    shape = (1, config.num_key_value_heads, length, config.hidden_size // config.num_attention_heads)
    if len(pairs) != config.num_hidden_layers:
        raise ValueError('Native cache does not contain every model layer')
    for key, value in pairs:
        if any(t.dtype != torch.float32 or tuple(t.shape) != shape for t in (key, value)):
            raise ValueError('Native cache FP32/GQA/position axis changed')
    return pairs


def cache_digest(cache):
    """Stream one tensor to host at a time; no whole-cache host private copy."""
    digest = hashlib.sha256()
    for pair in cache.to_legacy_cache():
        for tensor in pair:
            part = tensor.detach().contiguous().cpu().numpy()
            digest.update(str((part.shape, part.dtype)).encode())
            digest.update(memoryview(part).cast('B'))
            del part
    return digest.hexdigest()


def clone_cache(cache, config, length):
    """Clone all native postRoPE KV layers and establish storage independence."""
    import torch
    from transformers.cache_utils import DynamicCache
    pairs = _cache_tensors(cache, config, length)
    with torch.inference_mode():
        copied = tuple(tuple(value.clone() for value in pair) for pair in pairs)
        original_ptrs = {value.untyped_storage().data_ptr() for pair in pairs for value in pair}
        clone_ptrs = [value.untyped_storage().data_ptr() for pair in copied for value in pair]
        if len(set(clone_ptrs)) != len(clone_ptrs) or original_ptrs.intersection(clone_ptrs):
            raise RuntimeError('Cloned cache aliases original or another cloned tensor')
        if not all(torch.equal(a, b) for pair, copy in zip(pairs, copied, strict=True)
                   for a, b in zip(pair, copy, strict=True)):
            raise RuntimeError('Cache cloning changed native values')
        result = DynamicCache.from_legacy_cache(copied)
    return result


@contextmanager
def _noop(observer, cache, source_mask, layer_index):
    """Identical source postRoPE K/V and raw Q; never an efficacy intervention."""
    import torch
    pair = cache.to_legacy_cache()[layer_index]
    positions = torch.as_tensor(np.flatnonzero(source_mask), device=observer.device)
    if not len(positions):
        raise ValueError('No source positions for declared K/V identity control')
    before = cache_digest(cache)
    with torch.inference_mode():
        for value in pair:
            selected = value.index_select(2, positions).clone()
            value.index_copy_(2, positions, selected)
    if cache_digest(cache) != before:
        raise RuntimeError('Same-value source K/V replacement changed the cache')
    def identity(module, args, output):
        if output.dtype != torch.float32:
            raise RuntimeError('Native no-op Q is not FP32')
        return output.clone()
    handle = observer.layers[layer_index].self_attn.q_proj.register_forward_hook(identity)
    try:
        yield
    finally:
        handle.remove()


def allocator_cleanup(observer):
    """Release only unused own allocator blocks after residency has ended."""
    import torch
    report = observer.resource_report()
    if report['active_module'] is not None or not report['all_parameters_meta']:
        raise RuntimeError('Allocator cleanup requires no resident model weights')
    before = _resources(observer.device)
    if observer.device.type == 'cuda':
        torch.cuda.synchronize(observer.device)
        torch.cuda.empty_cache()
    return {'before': before, 'after': _resources(observer.device),
            'live_caches_preserved': True}


def qualify_sequence(observer, prompt, target, source_mask, *, layer_index=15,
                     observe=None, counts=None):
    """Execute exactly four native model calls and three unchunked head calls.

    Tiny CPU engineering tests call the same function with their tiny official
    observer.  Production run admits only its frozen original observer/config.
    """
    import torch
    counts = counts if counts is not None else {'model_calls': 0, 'head_calls': 0}
    config, length = observer.native.config, len(prompt)
    if len(source_mask) != length or length < 2 or not 0 <= target < config.vocab_size:
        raise ValueError('Invalid original prompt/source/target endpoint')
    if not 0 <= layer_index < config.num_hidden_layers:
        raise ValueError('Identity control layer is outside the native model')
    outputs, cleanups, resources = {}, {}, {}
    hooks_before = _hook_counts(observer.native)
    def model(ids, **kwargs):
        counts['model_calls'] += 1
        return observer.hidden(ids, **kwargs)
    def head(name, hidden):
        if hidden.dtype != torch.float32 or hidden.shape != (1, config.hidden_size):
            raise ValueError('Final native norm output is not the exact FP32 query row')
        cleanups[name] = {'before_head': allocator_cleanup(observer)}
        counts['head_calls'] += 1
        logits = observer.logits(hidden)
        if logits.dtype != torch.float32 or logits.shape != (1, config.vocab_size):
            raise ValueError('Unchunked native head dtype/full-vocabulary axis changed')
        logp = float(logits.log_softmax(-1)[0, target].item())
        result = {'hidden': hidden[0].detach().cpu().numpy().copy(),
                  'logits': logits[0].detach().cpu().numpy().copy(), 'logp': logp}
        del logits
        cleanups[name]['after_head'] = allocator_cleanup(observer)
        resources[name] = _resources(observer.device)
        outputs[name] = result
        if observe is not None:
            observe(name, result, counts, resources[name])
    with torch.inference_mode():
        precision = {'grad_enabled': torch.is_grad_enabled(),
                     'autocast_cpu': torch.is_autocast_enabled('cpu'),
                     'autocast_cuda': torch.is_autocast_enabled('cuda'),
                     'cuda_matmul_tf32': torch.backends.cuda.matmul.allow_tf32,
                     'cudnn_tf32': torch.backends.cudnn.allow_tf32,
                     'weight_dtypes': sorted({str(v.dtype) for v in observer.native.parameters()}),
                     'attention_backend': observer.native.config._attn_implementation}
        if (any(precision[name] for name in ('grad_enabled', 'autocast_cpu', 'autocast_cuda',
                                            'cuda_matmul_tf32', 'cudnn_tf32'))
                or precision['weight_dtypes'] != ['torch.bfloat16']
                or precision['attention_backend'] != 'sdpa'):
            raise RuntimeError('Full-head native precision contract changed')
        bulk = model(prompt, use_cache=False)
        head('bulk', bulk.last_hidden_state[0, -1:])
        del bulk
        prefix = model(prompt[:-1], use_cache=True)
        pristine = prefix.past_key_values
        del prefix
        _cache_tensors(pristine, config, length-1)
        pristine_sha = cache_digest(pristine)
        baseline_cache = clone_cache(pristine, config, length-1)
        noop_cache = clone_cache(pristine, config, length-1)
        a = {v.untyped_storage().data_ptr() for pair in baseline_cache.to_legacy_cache() for v in pair}
        b = {v.untyped_storage().data_ptr() for pair in noop_cache.to_legacy_cache() for v in pair}
        if a & b:
            raise RuntimeError('Independent cache branches share storage')
        baseline = model(prompt[-1:], past_key_values=baseline_cache, use_cache=True)
        head('cached', baseline.last_hidden_state[0, -1:])
        del baseline
        with _noop(observer, noop_cache, np.asarray(source_mask[:-1]), layer_index):
            changed = model(prompt[-1:], past_key_values=noop_cache, use_cache=True)
        head('noop', changed.last_hidden_state[0, -1:])
        del changed
        original_unchanged = (pristine.get_seq_length() == length-1
                              and cache_digest(pristine) == pristine_sha)
        for branch in (baseline_cache, noop_cache):
            _cache_tensors(branch, config, length)
        del pristine, baseline_cache, noop_cache
    comparison = compare_outputs(outputs['bulk'], outputs['cached'])
    noop = compare_outputs(outputs['cached'], outputs['noop'])
    clean = (_hook_counts(observer.native) == hooks_before
             and observer.resource_report()['all_parameters_meta']
             and observer.resource_report()['active_module'] is None)
    budget = counts == {'model_calls': 4, 'head_calls': 3}
    return {'qualified': comparison['within_budget'] and noop['bitwise_equal'] and original_unchanged and clean and budget,
            'bulk_vs_cached': comparison, 'cached_vs_noop': noop,
            'pristine_cache_unchanged': original_unchanged, 'all_layer_clone_storage_independent': True,
            'hook_and_residency_cleanup': clean, 'counts': dict(counts),
            'precision_flags': precision, 'call_budget_matched': budget,
            'allocator_cleanup': cleanups, 'resources': resources,
            'resource_report': observer.resource_report()}


def _prefix_prerequisite(root, sample, assets, observer_identity):
    seal = verify_seal(root, required=('plan.json', 'summary.json', 'native.json', 'blocked.json'))
    plan, summary = read_json(root/'plan.json'), read_json(root/'summary.json')
    if (summary.get('all_qualified') is not True or summary.get('model_calls') != 2
            or str(plan['id']) != sample or plan['assets'] != assets
            or plan['observer_capture_identity'] != observer_identity):
        raise ValueError('Sealed original-prefix qualification does not match this observer')
    prior = plan['source_runtime']['source_sha256']
    for key, path in [('experiments.research/streamed_observer.py', Path(__file__).parent/'research'/'streamed_observer.py'),
                      ('experiments/streamed_qualification.py', Path(__file__).with_name('streamed_qualification.py'))]:
        if prior.get(key) != file_hash(path):
            raise ValueError('Prefix-qualified shared execution dependency changed')
    return {'seal_sha256': file_hash(root/'seal.json'), 'members': seal,
            'complete_original_prompt_tokens': plan['complete_original_prompt_tokens'],
            'capture_sha256': plan['capture_sha256']}


def run(args):
    """Root-claimed native four-call gate; no new risk score or role QA.

    Required: capture_root, model, asset_manifest, prefix_qualification, output.
    Optional: asset_verification, asset_state, device='cuda:0', max_tokens=256,
    id='16465'.  Every original model/head call is counted, including attempts.
    """
    import torch
    started = time.perf_counter()
    capture, model, output = _path(args.capture_root), _path(args.model), _path(args.output)
    prefix_root = _path(args.prefix_qualification)
    sample = str(getattr(args, 'id', '16465'))
    if not sample.isdecimal():
        raise ValueError('Original answer identity must be decimal')
    assets = verify_assets(args.asset_manifest, model,
        verification=getattr(args, 'asset_verification', None), state=getattr(args, 'asset_state', None))
    observer_identity = capture_identity(capture, model/'config.json')
    prerequisite = _prefix_prerequisite(prefix_root, sample, assets, observer_identity)
    directory = capture/'capture'/sample
    record = read_json(directory/'record.json')
    maximum = getattr(args, 'max_tokens', 256)
    p, t, n = _validate_reference(archive_headers(directory/'arrays.npz'), record, maximum)
    if str(record['id']) != sample or p != prerequisite['complete_original_prompt_tokens']:
        raise ValueError('Full original prompt differs from prefix qualification')
    if any(file_hash(directory/name) != digest for name, digest in prerequisite['capture_sha256'].items()):
        raise ValueError('Original cached input bytes changed since prefix qualification')
    identity = research_identity()
    device, limits = torch.device(getattr(args, 'device', 'cuda:0')), ResourceLimits(max_sequence_tokens=maximum)
    plan = {'schema': 'h017-native-head-cache-noop-plan-v1', 'id': sample,
            'source_id': str(record['source_id']), 'complete_original_prompt_tokens': p,
            'query_position': p-1, 'target': 'original answer_ids[0]; raw observer temperature1',
            'planned_model_calls': 4, 'planned_unchunked_head_calls': 3,
            'policy': dict(POLICY), 'source_runtime': identity, 'assets': assets,
            'prefix_prerequisite': prerequisite, 'observer_capture_identity': observer_identity,
            'resource_limits': limits.__dict__, 'before': _resources(device),
            'allocator_policy': 'empty_cache before/after eachhead only with allweightsmeta/noactive_module',
            'cache': '32layers FP32 postRoPEK/V; independent pristine clones; source-only same-value KVno-op',
            'labels_read': False, 'detector_effectiveness_evaluated': False}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'plan.json', plan)
    counts, observer, observations = {'model_calls': 0, 'head_calls': 0}, None, []
    def save(name, result, actual_counts, resources):
        np.savez_compressed(output/f'{name}-output.npz', hidden=result['hidden'], logits=result['logits'])
        write_json(output/f'{name}.json', {'original_token_logp': result['logp'],
            'actual_counts': dict(actual_counts), 'resources': resources,
            'hidden_dtype': str(result['hidden'].dtype), 'logit_dtype': str(result['logits'].dtype)})
        observations.append(name)
    try:
        # Reuse exact axis validation; future response tokens are never model inputs.
        reference = _reference_payload(directory/'arrays.npz', p, t, n)
        prompt = reference['token_ids'][:p].tolist()
        target, source = int(reference['answer_ids'][0]), reference['source_mask'][:p]
        del reference
        store = SafeTensorStore(model, index_sha256=assets['files']['model.safetensors.index.json']['sha256'],
                                expected_files=assets['files'])
        observer = StreamedObserver(store, read_json(model/'config.json'),
            config_sha256=assets['files']['config.json']['sha256'], device=device, limits=limits)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        result = qualify_sequence(observer, prompt, target, source, observe=save, counts=counts)
        result.update(schema='h017-native-head-cache-noop-result-v1',
                      original_first_answer_token_id=target, full_model_completed=True,
                      observer_conditional_probability=True, original_generator_probability=False,
                      elapsed_seconds=time.perf_counter()-started, after=_resources(device),
                      detector_effectiveness_evaluated=False, labels_read=False)
        write_json(output/'result.json', result)
    except BaseException as error:
        write_json(output/'engineering-failure.json', {'type': type(error).__name__, 'message': str(error),
            'actual_counts': counts, 'observations_saved': observations, 'resources': _resources(device),
            'elapsed_seconds': time.perf_counter()-started, 'automatic_retry': False})
        raise
    finally:
        if observer is not None:
            observer.close()
    completed_identity(identity)
    if file_hash(prefix_root/'seal.json') != prerequisite['seal_sha256']:
        raise RuntimeError('Prefix qualification seal changed during head qualification')
    verify_seal(prefix_root, expected=prerequisite['members'])
    if verify_assets(args.asset_manifest, model, verification=getattr(args, 'asset_verification', None),
                     state=getattr(args, 'asset_state', None)) != assets:
        raise RuntimeError('Original asset identities changed during head qualification')
    write_json(output/'seal.json', {path.name: file_hash(path) for path in sorted(output.iterdir()) if path.is_file()})
    return result
