"""Qualify recovered physical-head attention coordinates; never score a response.

This instrument replays the frozen producer contract, not a detector.  The cache
contains FP16 *pre-RoPE* projections from FP32 observer execution.  Consequently
replay is approximate, and its rounding interval is part of every comparison.
Only Transformers' official matching RoPE initializer is imported: loading a
complete model implementation or model weights is unnecessary for this audit.
"""

from dataclasses import asdict, dataclass
from importlib.metadata import version
import zipfile

import numpy as np


GROUPS = ('source', 'local', 'remote', 'self', 'other_prompt')
FIELDS = {'nodes', 'query', 'key', 'value', 'local_attention', 'group_heads',
          'group_mass', 'token_ids', 'answer_ids', 'source_mask',
          'query_positions', 'local_positions', 'local_valid'}
READ_FIELDS = ('query', 'key', 'local_attention', 'group_mass', 'token_ids',
               'answer_ids', 'source_mask', 'query_positions',
               'local_positions', 'local_valid')
CONTROLS = ('omit_rope', 'shift_query_row', 'interleave_gqa',
            'omit_causal_mask', 'omit_source_mask')


@dataclass(frozen=True)
class ReplayPolicy:
    """Frozen engineering tolerances, independent of labels or real canaries.

For normal/subnormal FP16 rounding, |x-x16| <= u|x16|/(1-u)+2^-25,
u=2^-11.  Absolute RoPE and bilinear QK propagate that interval.  If every
logit changes by at most delta, the probability of *any* key subset changes
by at most tanh(delta/2), including singleton local-key probabilities.

The additional FP32 allowance documents a conservative cross-device arithmetic
budget, not a universal proof about CUDA/CPU libm.  Large intervals are unknown
rather than evidence of a qualified instrument.
"""

    fp32_logit_absolute: float = 2e-4
    # 1024*eps32 exceeds the two width128 dot-product gamma_128 budgets plus
    # elementary FP32 rotations; the absolute allowance covers backend/libm.
    fp32_dot_factor: float = 1024.
    mass_absolute: float = 1e-5
    precision_ceiling: float = .05
    transformers_version: str = '4.57.1'

    def to_dict(self):
        return asdict(self)


DEFAULT_POLICY = ReplayPolicy()


def archive_headers(path):
    """Read only NPY headers; reject unexpected members and object arrays."""
    result = {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(FIELDS) or set(names) != {f'{n}.npy' for n in FIELDS}:
            raise ValueError('Unexpected capture archive fields')
        for name in sorted(FIELDS):
            with archive.open(f'{name}.npy') as stream:
                major, minor = np.lib.format.read_magic(stream)
                if (major, minor) == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif (major, minor) == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('Unsupported capture NPY version')
            if dtype.hasobject or fortran:
                raise ValueError('Capture must contain plain C-order arrays')
            size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
            result[name] = {'shape': list(shape), 'dtype': str(dtype), 'bytes': size}
    return result


def memory_requirement(headers, query_chunk):
    """Conservative incremental working-set admission, excluding Python runtime.

    Q/K and canaries are the only decoded archive fields.  The estimate includes
    their decompression copies, float32 rotations/error envelopes, and a bounded
    query chunk of logits, probabilities, masks and error products.  It never
    admits a full [all queries, all keys, all heads] tensor.
    """
    if query_chunk < 1:
        raise ValueError('query_chunk must be positive')
    raw = sum(headers[n]['bytes'] for n in READ_FIELDS)
    worlds, senders, kv_heads, width = headers['key']['shape']
    heads = headers['query']['shape'][2]
    key_work = senders * kv_heads * width * 4 * 12
    # GQA expansion, float32 key rotations, float64 interval workspace.
    expanded_key = senders * heads * width * 8 * 4
    chunk_work = query_chunk * senders * heads * 8 * 10
    return int(2 * raw + key_work + expanded_key + chunk_work + 16 * 1024**2)


def load_capture(path, *, query_chunk=8, max_memory_bytes=512 * 1024**2):
    """Decode one admitted archive, omitting nodes, V and AV message payloads."""
    headers = archive_headers(path)
    required = memory_requirement(headers, query_chunk)
    if required > max_memory_bytes:
        raise MemoryError(f'Capture requires estimated {required} incremental bytes; '
                          f'budget is {max_memory_bytes}')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in READ_FIELDS}
    return arrays, {'headers': headers, 'estimated_incremental_bytes': required}


def validate_axes(arrays, record, response=None):
    """Validate the complete original T+1 axis before interpreting coordinates.

    The optional prepared response is validated through the mainline answer_axis;
    without it no original character-axis validation is claimed.
    """
    audit = record['audit']
    tokens = arrays['token_ids']
    answer = arrays['answer_ids']
    n, t = len(tokens), len(answer)
    prompt = n - t
    if prompt < 1 or t < 1:
        raise ValueError('Empty prompt/answer capture is unsupported')
    rows, heads, kv, width = t + 1, 32, 8, 128
    shapes = {'query': (2, rows, heads, width), 'key': (2, n, kv, width),
              'local_attention': (2, rows, heads, 8),
              'group_mass': (2, rows, 5, heads), 'source_mask': (n,),
              'query_positions': (rows,), 'local_positions': (rows, 8),
              'local_valid': (rows, 8)}
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f'Capture {name} shape differs from producer contract')
    for name in ('query', 'key', 'local_attention'):
        if arrays[name].dtype != np.float16:
            raise ValueError(f'Capture {name} is not archived FP16')
    if arrays['group_mass'].dtype != np.float32:
        raise ValueError('Capture group_mass is not FP32')
    for name in ('source_mask', 'local_valid'):
        if arrays[name].dtype != np.bool_:
            raise ValueError(f'Capture {name} is not boolean')
    for name in ('token_ids', 'answer_ids', 'query_positions', 'local_positions'):
        if arrays[name].dtype != np.int64:
            raise ValueError(f'Capture {name} does not preserve int64 token axes')
    positions = np.arange(prompt - 1, n, dtype=np.int64)
    neighbors = positions[:, None] - np.arange(1, 9)
    if (not np.array_equal(tokens[prompt:], answer)
            or not np.array_equal(arrays['query_positions'], positions)
            or not np.array_equal(arrays['local_positions'], neighbors)
            or not np.array_equal(arrays['local_valid'], neighbors >= prompt)
            or arrays['source_mask'][prompt:].any()):
        raise ValueError('Original token, source, query or strict-past local axis changed')
    expected = {'layer': 15, 'local_width': 8, 'input_tokens': n,
                'prompt_length': prompt, 'answer_tokens': t, 'node_rows': rows,
                'stop_after_layer': True, 'full_model_forwards': 0}
    if any(audit.get(k) != v for k, v in expected.items()):
        raise ValueError('Record metadata differs from selected-layer early-stop contract')
    worlds = audit.get('worlds', [])
    if (len(worlds) != 2 or [w.get('blocked') for w in worlds] != [False, True]
            or any(w.get('executed_layers') != 16 or w.get('full_model_completed') is not False
                   for w in worlds) or tuple(audit.get('groups', [])) != GROUPS):
        raise ValueError('World or partition metadata differs from producer contract')
    for name in ('query', 'key', 'local_attention', 'group_mass'):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f'Non-finite recovered {name}')
    if (np.any(arrays['local_attention'] < 0) or np.any(arrays['local_attention'] > 1)
            or np.any(arrays['group_mass'] < 0) or np.any(arrays['group_mass'] > 1)):
        raise ValueError('Invalid saved probability canary')
    if not np.allclose(arrays['group_mass'].sum(axis=2), 1., atol=1e-5, rtol=0):
        raise ValueError('Saved partition masses do not conserve probability')
    if np.any(arrays['local_attention'] * ~arrays['local_valid'][None, :, None]):
        raise ValueError('Saved invalid local edges are nonzero')
    result = {'input_tokens': n, 'answer_tokens': t, 'prompt_tokens': prompt,
              'query_rows': rows, 'worlds': 2, 'physical_heads': heads,
              'prechoice_rows': [0, t], 'posttoken_rows': [1, rows],
              'character_axis_validated': response is not None,
              'full_model_probability_available': False,
              'original_generator_internal_state': False}
    if response is not None:
        from myproject.measure import answer_axis
        if not np.array_equal(response['answer_ids'], answer):
            raise ValueError('Prepared original response token IDs differ')
        axis = answer_axis(response)
        result['original_evaluation_tokens'] = len(axis['target'])
    return result


def rope_tables(config, positions, policy=DEFAULT_POLICY):
    """Use the official Llama3 initializer, with original absolute positions.

    Only the initializer is model-specific.  The elementary cos/sin application
    below matches the archived producer's documented preRoPE coordinate contract.
    """
    import torch
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.models.llama.configuration_llama import LlamaConfig
    if version('transformers') != policy.transformers_version:
        raise ValueError('Replay requires the captured Transformers version')
    if (config.get('model_type') != 'llama' or config.get('num_attention_heads') != 32
            or config.get('num_key_value_heads') != 8 or config.get('hidden_size') != 4096
            or config.get('rope_scaling', {}).get('rope_type') != 'llama3'):
        raise ValueError('Unsupported observer architecture/RoPE contract')
    native = LlamaConfig.from_dict(config)
    inverse, amplitude = ROPE_INIT_FUNCTIONS['llama3'](native, torch.device('cpu'))
    angles = torch.as_tensor(positions, dtype=torch.float32)[:, None] * inverse[None]
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos() * amplitude, angles.sin() * amplitude


def _half_turn(values):
    left, right = values.chunk(2, dim=-1)
    import torch
    return torch.cat((-right, left), dim=-1)


def _rotate(values, cosine, sine):
    return values * cosine[:, None] + _half_turn(values) * sine[:, None]


def _fp16_interval(values):
    return values.abs().double() * (2.**-11 / (1 - 2.**-11)) + 2.**-25


def _rotation_interval(values, cosine, sine):
    interval = _fp16_interval(values)
    return interval * cosine[:, None].abs().double() + _half_turn(interval).abs() * sine[:, None].abs().double()


def sender_groups(source, positions, prompt, width=8):
    """Producer partition masks; source/answer/other/self remain distinct."""
    import torch
    keys = torch.arange(len(source))
    past = keys[None] < positions[:, None]
    source_keys = past & source[None]
    history = past & (keys[None] >= prompt) & ~source[None]
    local = history & (positions[:, None] - keys[None] <= width)
    return torch.stack((source_keys, local, history & ~local,
                        keys[None] == positions[:, None],
                        past & (keys[None] < prompt) & ~source[None]), dim=1)


def iter_attention(query, key, positions, source_mask, config, *, blocked,
                   query_chunk=8, variant='faithful', policy=DEFAULT_POLICY,
                   with_bounds=True):
    """Yield bounded CPU query chunks, preserving all physical heads.

    Input axes: Q[rows,32,128], K[keys,8,128].  Output attention axes are
    [rows,32,keys].  Wrong-contract variants are engineering negative controls,
    never detector ablations or alternative method implementations.
    """
    import torch
    if variant not in ('faithful', *CONTROLS):
        raise ValueError('Unknown replay variant')
    if query_chunk < 1:
        raise ValueError('query_chunk must be positive')
    qraw = torch.from_numpy(query.astype(np.float32))
    kraw = torch.from_numpy(key.astype(np.float32))
    pos = torch.as_tensor(positions, dtype=torch.long)
    source = torch.as_tensor(source_mask, dtype=torch.bool)
    cosine, sine = rope_tables(config, np.arange(len(key)), policy)
    rotated_key = _rotate(kraw, cosine, sine)
    key_error = _rotation_interval(kraw, cosine, sine) if with_bounds else None
    # Contiguous repetition is native GQA: heads 0..3 share KV head0, etc.
    mapping = torch.arange(32) // 4
    if variant == 'interleave_gqa':
        mapping = torch.arange(32) % 8
    if variant == 'omit_rope':
        rotated_key = kraw
    key_heads = rotated_key[:, mapping].permute(1, 2, 0).contiguous()
    error_heads = key_error[:, mapping].permute(1, 2, 0).contiguous() if with_bounds else None
    if variant == 'shift_query_row':
        qraw = torch.roll(qraw, shifts=1, dims=0)
    scale = query.shape[-1]**-.5
    keys = torch.arange(len(key))
    for begin in range(0, len(query), query_chunk):
        end = min(begin + query_chunk, len(query))
        rows = pos[begin:end]
        raw = qraw[begin:end]
        rotated_query = _rotate(raw, cosine[rows], sine[rows])
        if variant == 'omit_rope':
            rotated_query = raw
        qheads = rotated_query.permute(1, 0, 2)
        logits = torch.matmul(qheads, key_heads) * scale
        allowed = keys[None] <= rows[:, None]
        if variant == 'omit_causal_mask':
            allowed = torch.ones_like(allowed)
        if blocked and variant != 'omit_source_mask':
            allowed &= source[rows, None] | ~source[None]
        if not allowed.any(dim=1).all():
            raise ValueError('Replay query has no allowed key')
        attention = logits.masked_fill(~allowed[None], -torch.inf).softmax(-1).permute(1, 0, 2)
        bound = None
        if with_bounds:
            qerror = _rotation_interval(raw, cosine[rows], sine[rows]).permute(1, 0, 2)
            qabs, kabs = qheads.abs().double(), key_heads.abs().double()
            delta = (torch.matmul(qerror, kabs) + torch.matmul(qabs, error_heads)
                     + torch.matmul(qerror, error_heads)) * scale
            delta += (policy.fp32_dot_factor * np.finfo(np.float32).eps
                      * torch.matmul(qabs, kabs) * scale + policy.fp32_logit_absolute)
            delta = delta.masked_fill(~allowed[None], 0.).amax(dim=-1).transpose(0, 1)
            bound = torch.tanh(delta / 2).numpy() + policy.mass_absolute
        yield begin, end, attention, allowed, bound


def _empty_metric(*, per_head=True):
    metric = {'elements': 0, 'eligible_elements': 0, 'precision_unknown_elements': 0,
            'violations': 0, 'max_absolute_error': 0., 'max_tolerance': 0.,
            'max_error_over_tolerance': 0.}
    if per_head:
        metric['physical_heads'] = [_empty_metric(per_head=False) for _ in range(32)]
    return metric


def _compare(metric, actual, expected, bound, policy, *, fp16=False, head_axis=None):
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    tolerance = np.broadcast_to(bound, error.shape).copy()
    if fp16:
        tolerance += _fp16_probability_interval(expected)
    eligible = tolerance <= policy.precision_ceiling
    metric['elements'] += error.size
    metric['eligible_elements'] += int(eligible.sum())
    metric['precision_unknown_elements'] += int((~eligible).sum())
    metric['violations'] += int(((error > tolerance) & eligible).sum())
    metric['max_absolute_error'] = max(metric['max_absolute_error'], float(error.max(initial=0)))
    metric['max_tolerance'] = max(metric['max_tolerance'], float(tolerance.max(initial=0)))
    metric['max_error_over_tolerance'] = max(metric['max_error_over_tolerance'],
        float(np.max(np.divide(error, tolerance, where=eligible,
                               out=np.zeros_like(error)), initial=0)))
    if head_axis is not None:
        for head in range(32):
            # Same already-fixed interval, without adding canary rounding twice.
            _compare(metric['physical_heads'][head], np.take(actual, head, axis=head_axis),
                     np.take(expected, head, axis=head_axis),
                     np.take(tolerance, head, axis=head_axis), policy)


def _fp16_probability_interval(values):
    return np.abs(values.astype(np.float64)) * (2.**-11 / (1 - 2.**-11)) + 2.**-25


def qualify_arrays(arrays, config, record, *, query_chunk=8, policy=DEFAULT_POLICY,
                   include_controls=True, response=None):
    """Compare saved canaries with faithful replay and wrong-contract controls.

    No annotation, correctness label, risk score, head selection or model forward
    is accepted by this interface.  Controls retain the faithful implementation's
    tolerance; their wrong inputs must not widen their own acceptance intervals.
    """
    import torch
    axis = validate_axes(arrays, record, response)
    variants = ('faithful', *CONTROLS) if include_controls else ('faithful',)
    metrics = {v: {'local_attention': _empty_metric(), 'group_mass': _empty_metric()}
               for v in variants}
    source = torch.as_tensor(arrays['source_mask'])
    positions = torch.as_tensor(arrays['query_positions'])
    bounds = []
    for world in range(2):
        world_bounds = []
        for begin, end, attention, allowed, bound in iter_attention(
                arrays['query'][world], arrays['key'][world], arrays['query_positions'],
                arrays['source_mask'], config, blocked=bool(world), query_chunk=query_chunk,
                policy=policy):
            world_bounds.append(bound)
            _compare_chunk(metrics['faithful'], arrays, world, begin, end, attention,
                           bound, axis['prompt_tokens'], source, positions, policy)
            forbidden = (attention.masked_select(~allowed[:, None].expand_as(attention))
                         .abs().max().item() if (~allowed).any() else 0.)
            if forbidden:
                raise ValueError('Forbidden reconstructed attention is nonzero')
        bounds.append(np.concatenate(world_bounds))
    for variant in variants[1:]:
        for world in range(2):
            if variant == 'omit_source_mask' and world == 0:
                continue  # Identical native world carries no mask-control information.
            for begin, end, attention, _, _ in iter_attention(
                    arrays['query'][world], arrays['key'][world], arrays['query_positions'],
                    arrays['source_mask'], config, blocked=bool(world), query_chunk=query_chunk,
                    variant=variant, policy=policy, with_bounds=False):
                _compare_chunk(metrics[variant], arrays, world, begin, end, attention,
                               bounds[world][begin:end], axis['prompt_tokens'],
                               source, positions, policy)
    faithful = metrics['faithful']
    violations = sum(m['violations'] for m in faithful.values())
    unknown = sum(m['precision_unknown_elements'] for m in faithful.values())
    controls = {v: ('rejected' if sum(m['violations'] for m in metrics[v].values()) else
                    'non_identifying') for v in variants[1:]}
    status = ('numeric_contract_failed' if violations else
              'precision_unknown' if unknown else
              'negative_controls_non_identifying' if any(s != 'rejected' for s in controls.values()) else
              'qualified' if include_controls else 'faithful_only_controls_not_run')
    return {'status': status, 'axis': axis, 'policy': policy.to_dict(),
            'metrics': metrics, 'controls': controls,
            'scope': 'FP16 archived observer attention identity only; no detector effectiveness'}


def _compare_chunk(metrics, arrays, world, begin, end, attention, bound,
                   prompt, source, positions, policy):
    import torch
    local = torch.as_tensor(arrays['local_positions'][begin:end]).clamp_min(0)
    local_weights = attention.gather(2, local[:, None].expand(-1, 32, -1))
    local_weights *= torch.as_tensor(arrays['local_valid'][begin:end])[:, None]
    groups = sender_groups(source, positions[begin:end], prompt)
    mass = torch.einsum('rhs,rgs->rgh', attention, groups.to(attention.dtype))
    _compare(metrics['local_attention'], local_weights.numpy(),
             arrays['local_attention'][world, begin:end], bound[:, :, None], policy,
             fp16=True, head_axis=1)
    _compare(metrics['group_mass'], mass.numpy(),
             arrays['group_mass'][world, begin:end], bound[:, None, :], policy, head_axis=2)
