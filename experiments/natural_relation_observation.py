"""One claimed, unmodified natural-response observation, never a detector.

The source number's K/V cannot depend on a later observed BC marker, although
left context and priors may predict it. Later states can access that marker;
access, encoding, use and causal necessity remain different questions.
"""

from contextlib import ExitStack
from pathlib import Path
import json
import re
import time

import numpy as np

from .data import file_hash, read_json, text_hash, verify_seal, write_json
from .datasets import Sample
from .provenance import completed_identity, research_identity
from .relation_alignment import _exact_offsets, reconstruct_alignment
from .relation_cache_qualification import capture_identity
from .relation_intervention import _arrays, _seal, _save_cache, head_prerequisite
from .research.relation_intervention import AV_POLICY, attention_diagnostic, capture_rows
from .research.relation_inventory import CONTRACT
from .research.streamed_observer import ResourceLimits, SafeTensorStore, StreamedObserver
from .streamed_head_qualification import allocator_cleanup
from .streamed_qualification import (
    COORDINATE_POLICY, _hook_counts, _native_dtype_guards, _path,
    _reference_payload, _resources, coordinate_summary, frozen_capture_functions, verify_assets,
)


ADMISSION_SHA256 = '901c677ff845ba3a137f02a9ce0650c2a86d228392fd68c09e243af44423eb58'
LICENSE_SHA256 = '23e77b9ac9cb2703b14e49cd61b62a910f008ab865e3adce18e7e8e03f636f0a'
CASE, SOURCE, LENGTHS = '15716', '15540', (365, 76, 441)
ANCHORS = {
    'G03': (138, 244), 'time_scope': (226, 243), 'number300': (231, 234),
    'eraBC': (241, 243), 'G03_closure': (243, 244),
    'G06': (550, 593), 'angle30': (560, 562),
}
LITERALS = {'time_scope': 'over 300 years BC', 'number300': '300',
            'eraBC': 'BC', 'G03_closure': '.', 'angle30': '30'}


class _RecordingTokenizer:
    """Retain offsets produced inside the unchanged mainline M2 encoder."""
    def __init__(self, tokenizer):
        self.tokenizer, self.calls = tokenizer, []

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, text, **kwargs):
        encoded = self.tokenizer(text, **kwargs)
        self.calls.append((text, encoded['input_ids'], encoded['offset_mapping']))
        return encoded


def prepare_axes(sample, tokenizer, axes, previous):
    """Reuse M2 verbatim, retaining manual-anchor overlap and original offsets."""
    inventory = dict(schema=CONTRACT, source_sha256=text_hash(sample.source),
        fields=[dict(occurrence_id=name, span=list(span), raw_text=sample.source[slice(*span)])
                for name, span in ANCHORS.items()], lexical_occurrences=[])
    for name, literal in LITERALS.items():
        if sample.source[slice(*ANCHORS[name])] != literal:
            raise ValueError('Frozen source anchor literal changed')
    recorded = _RecordingTokenizer(tokenizer)
    aligned = reconstruct_alignment(sample, recorded, axes, inventory)
    for key in ('prompt_tokens', 'answer_tokens', 'full_tokens', 'query_rows',
                'prompt_character_count', 'response_sha256', 'id', 'source_id'):
        expected = {'id': sample.id, 'source_id': sample.source_id}.get(key, aligned.get(key))
        if previous.get(key) != expected:
            raise ValueError('Original M2 case/count/response identity changed')
    for key in ('source_sha256', 'token_ids_sha256', 'offsets_sha256', 'source_start', 'tokens'):
        if previous['alignment'].get(key) != aligned['alignment'].get(key):
            raise ValueError('Original M2 token/character axis changed')
    p = aligned['prompt_tokens']
    prompts = [row for row in recorded.calls if row[1] == axes['token_ids'][:p]]
    answers = [row for row in recorded.calls if row[0] == sample.response
               and row[1] == axes['answer_ids']]
    if not prompts or len(answers) != 1 or any(row != prompts[0] for row in prompts):
        raise ValueError('Mainline tokenizer offset observations are ambiguous')
    prompt, response = prompts[0], answers[0]
    offsets = _exact_offsets(prompt[2], len(prompt[0]))
    answer_offsets = _exact_offsets(response[2], len(response[0]))
    offsets += _exact_offsets(response[2], len(response[0]), len(prompt[0]))
    mapping = {}
    for occurrence in aligned['alignment']['occurrences']:
        name = occurrence['occurrence_id']
        mapping[name] = dict(occurrence, literal=inventory['fields'][list(ANCHORS).index(name)]['raw_text'],
            tokens=[dict(position=i, token_id=axes['token_ids'][i], model_offset=offsets[i])
                    for i in occurrence['overlap_tokens']])
    era, number = mapping['eraBC']['overlap_tokens'], mapping['number300']['overlap_tokens']
    frontier_known = bool(era and number and max(number) < min(era))
    response_anchors = {}
    for word in ('300', 'years', 'ago'):
        response_anchors[word] = [dict(character_span=[match.start(), match.end()],
            answer_indices=[i for i, (start, stop) in enumerate(answer_offsets)
                            if start < match.end() and stop > match.start()])
            for match in re.finditer(r'\b' + re.escape(word) + r'\b', sample.response)]
    return dict(alignment=aligned, source_anchors=mapping, response_anchors=response_anchors,
        full_offsets=offsets, answer_offsets=answer_offsets,
        era_frontier_status='known_token_order' if frontier_known else 'unknown',
        era_first_token=min(era) if frontier_known else None,
        era_complete_token=max(era) if frontier_known else None,
        semantic_frontend='manual_source_license_not_automatic', source_reading_Q_observed=False)


def token_endpoints(logits, answer_ids, offsets):
    """Absolute full-vocabulary probabilities for the supplied original targets."""
    import torch
    if (logits.dtype != torch.float32 or logits.ndim != 2
            or len(answer_ids) != logits.shape[0] or len(offsets) != len(answer_ids)
            or not torch.isfinite(logits).all()):
        raise ValueError('Original-token full-vocabulary output axis changed')
    if any(type(i) is not int or not 0 <= i < logits.shape[1] for i in answer_ids):
        raise ValueError('Original answer IDs are not exact vocabulary indices')
    logp = logits.log_softmax(-1)
    result = []
    for index, token in enumerate(answer_ids):
        value = logp[index, token]
        result.append(dict(answer_index=index, token_id=token, character_span=offsets[index],
            logit=float(logits[index, token]), logp=float(value), probability=float(value.exp()),
            probability_above_half=bool(value.exp() > .5),
            rank=int((logits[index] > logits[index, token]).sum()) + 1,
            top1=int(logits[index].argmax()),
            entropy=float(-(logp[index].exp() * logp[index]).sum())))
    return result


def query_axis(axes, answer_offsets):
    """Separate the token at a receiver from the original next-token target."""
    ids, answer, rows = axes['token_ids'], axes['answer_ids'], axes['query_positions']
    p, t = len(ids) - len(answer), len(answer)
    if (p < 1 or len(answer_offsets) != t or rows != list(range(p - 1, len(ids)))
            or ids[p:] != answer or len(axes['source_mask']) != len(ids)
            or any(type(i) is not int for i in ids + answer + rows)
            or any(type(i) is not bool for i in axes['source_mask'])
            or any(axes['source_mask'][p:])):
        raise ValueError('Original full-token and T+1/target axes changed')
    return [dict(query_row=i, absolute_query_position=position, receiver_token_id=ids[position],
        receiver_kind='prompt_last' if i == 0 else 'original_response_token',
        receiver_answer_index=None if i == 0 else i - 1,
        receiver_character_span=None if i == 0 else answer_offsets[i - 1],
        target_answer_index=i if i < t else None, target_token_id=answer[i] if i < t else None,
        target_character_span=answer_offsets[i] if i < t else None,
        observed_next_target=i < t) for i, position in enumerate(rows)]


def profiles(diagnostic, axes, mapping):
    """Unconditional per-physical-head masses; overlapping anchors stay separate."""
    attention = diagnostic['attention']
    heads, rows, length = attention.shape
    positions = np.asarray(axes['query_positions'], dtype=np.int64)
    source = np.asarray(axes['source_mask'], dtype=bool)
    if rows != len(positions) or length != len(source):
        raise ValueError('Native attention and original receiver/key axes differ')
    result = {name: attention[..., row['overlap_tokens']].sum(-1)
              for name, row in mapping['source_anchors'].items()}
    result['source'] = attention[..., source].sum(-1)
    p = len(axes['token_ids']) - len(axes['answer_ids'])
    key = np.arange(length)[None, :]
    result['other_prompt'] = attention[..., (~source & (np.arange(length) < p))].sum(-1)
    result['earlier_R'] = (attention * ((key >= p) & (key < positions[:, None]))[None]).sum(-1)
    result['self'] = attention[np.arange(heads)[:, None], np.arange(rows)[None, :], positions[None, :]]
    if mapping['era_frontier_status'] == 'known_token_order':
        for name, boundary in (('source_at_or_after_era_first', mapping['era_first_token']),
                               ('source_at_or_after_era_complete', mapping['era_complete_token'])):
            result[name] = attention[..., source & (np.arange(length) >= boundary)].sum(-1)
    return result


def native_archive_gate(captured, reference):
    observed = dict(nodes=np.stack([captured[name].numpy() for name in
        ('residual', 'attention_write', 'mlp_write')], axis=1),
        query=captured['query'].numpy().reshape(reference['query'].shape[1:]),
        key=captured['key'].numpy().reshape(reference['key'].shape[1:]),
        value=captured['value'].numpy().reshape(reference['value'].shape[1:]))
    metrics = {name: coordinate_summary(value, reference[name][0],
        head_axis=None if name == 'nodes' else -2) for name, value in observed.items()}
    return dict(qualified=all(row['violations'] == 0 for row in metrics.values()), metrics=metrics,
                native_world_only=True, coordinate_policy=dict(COORDINATE_POLICY))


def observe_original(observer, axes, reference, mapping, capture, output, *, counts=None,
                     archival_layer=15, dtype_guard=_native_dtype_guards):
    """The sole full forward and sole untiled head; synthetic CPU callers share it."""
    import torch
    output = Path(output)
    if (output / 'raw').exists():
        raise FileExistsError('Existing native data is retained, never automatically restarted')
    counts = counts if counts is not None else {'model_calls': 0, 'head_calls': 0}
    if counts != {'model_calls': 0, 'head_calls': 0}:
        raise ValueError('A natural observation may start only with zero prior attempts')
    raw = output / 'raw'
    raw.mkdir(exist_ok=False)
    rows, t = axes['query_positions'], len(axes['answer_ids'])
    n, p = len(axes['token_ids']), len(axes['token_ids']) - t
    receiver_target_axis = query_axis(axes, mapping['answer_offsets'])
    write_json(raw / 'query-axis.json', dict(rows=receiver_target_axis,
        interpretation='Post R[i-1] and pre R[i] are the same receiver; final row has no next target'))
    hooks = _hook_counts(observer.native)
    traces, captured, native_dtypes, native, precision = {}, {}, {}, None, {}
    try:
        with torch.inference_mode(), ExitStack() as stack:
            precision = dict(inference_mode=torch.is_inference_mode_enabled(),
                grad_enabled=torch.is_grad_enabled(), autocast_cpu=torch.is_autocast_enabled('cpu'),
                autocast_cuda=torch.is_autocast_enabled('cuda'),
                cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                weight_parameter_dtypes=sorted({str(v.dtype) for v in observer.native.parameters()}),
                rotary_inverse_frequency_dtype=str(observer.native.model.rotary_emb.inv_freq.dtype))
            if (precision['grad_enabled'] or precision['autocast_cpu'] or precision['autocast_cuda']
                    or precision['cuda_matmul_tf32'] or precision['cudnn_tf32']
                    or precision['weight_parameter_dtypes'] != ['torch.bfloat16']
                    or precision['rotary_inverse_frequency_dtype'] != 'torch.float32'):
                raise RuntimeError('Original natural-observation precision context changed')
            native_dtypes = stack.enter_context(dtype_guard(observer.native))
            for layer in range(len(observer.layers)):
                traces[layer] = stack.enter_context(capture_rows(observer, rows, layer=layer))
            captured = stack.enter_context(capture(observer.native, archival_layer, p - 1,
                                                   stop_after_layer=False))
            counts['model_calls'] += 1
            write_json(output / 'model-attempt.json', dict(actual_counts=dict(counts),
                full_input_tokens=n, query_rows=len(rows), automatic_retry=False))
            native = observer.hidden(axes['token_ids'], use_cache=True)
    finally:
        # Completed or partial native data is retained before numerical rejection.
        _arrays(raw / 'axes.npz', {key: np.asarray(value, dtype=bool if key == 'source_mask' else np.int64)
                                 for key, value in axes.items()})
        for layer, trace in traces.items():
            _arrays(raw / f'layer-{layer:02d}.npz', {key: value.numpy() for key, value in trace.items()})
        _arrays(raw / 'archive-layer.npz', {key: value.numpy() for key, value in captured.items()})
        if native is not None:
            _save_cache(raw / 'all-layer-KV.npz', native.past_key_values)
            _arrays(raw / 'final-hidden.npz', {'hidden': native.last_hidden_state[0].detach().cpu().numpy()})
        write_json(raw / 'receipt.json', dict(actual_counts=dict(counts),
            forward_completed=native is not None, observed_layers=list(traces),
            native_dtypes_before_capture_cast=native_dtypes, precision_flags=precision,
            hook_cleanup=_hook_counts(observer.native) == hooks))
        _seal(raw)
    archive = native_archive_gate(captured, reference)
    write_json(output / 'native-archive-gate.json', archive)
    if not archive['qualified']:
        raise RuntimeError('Native full input does not reproduce every archived layer15 coordinate')
    pairs = native.past_key_values.to_legacy_cache()
    if (len(pairs) != len(observer.layers) or any(len(pair) != 2 or any(
            value.dtype != torch.float32 or tuple(value.shape) != (1, observer.head_layout(layer)[1],
                n, observer.head_layout(layer)[2]) or not torch.isfinite(value).all()
            for value in pair) for layer, pair in enumerate(pairs))):
        raise RuntimeError('Full native post-RoPE all-layer cache dtype/shape changed')
    phase, phase_root = [], output / 'attention'
    phase_root.mkdir(exist_ok=False)
    write_json(phase_root / 'profile-basis.json', dict(
        basis='Unconditional attention over original absolute source/key positions, per physical head and query row',
        anchor_intervals_may_overlap=True, anchor_profile_is_not_a_partition=True,
        era_frontier_status=mapping['era_frontier_status'],
        source_reading_Q_observed=False, no_semantic_encoding_or_use_claim=True))
    for layer, trace in traces.items():
        diagnostic = attention_diagnostic(observer, trace, native.past_key_values, rows, layer=layer)
        _arrays(phase_root / f'layer-{layer:02d}.npz', {key: value for key, value in diagnostic.items()
                                                    if key not in {'key', 'value'}})
        _arrays(phase_root / f'profile-{layer:02d}.npz', profiles(diagnostic, axes, mapping))
        receipt = dict(layer=layer, physical_heads=diagnostic['attention'].shape[0],
            receivers=len(rows), keys=n, av_max_error=diagnostic['av_max_error'],
            av_max_error_over_tolerance=diagnostic['av_max_error_over_tolerance'])
        write_json(phase_root / f'layer-{layer:02d}.json', receipt)
        phase.append(receipt)
        del diagnostic
    _seal(phase_root)
    selected = native.last_hidden_state[:, rows[:-1], :]
    if (tuple(selected.shape) != (1, t, observer.native.config.hidden_size)
            or selected.dtype != torch.float32 or not torch.isfinite(selected).all()):
        raise RuntimeError('Original 76 prechoice head rows changed')
    allocator_cleanup(observer)
    counts['head_calls'] += 1
    write_json(output / 'head-attempt.json', dict(actual_counts=dict(counts),
        rows=t, vocabulary=observer.native.config.vocab_size, unchunked=True, automatic_retry=False))
    logits = observer.logits(selected)[0].detach().cpu()
    _arrays(output / 'output.npz', dict(hidden=selected[0].detach().cpu().numpy(), logits=logits.numpy()))
    if logits.dtype != torch.float32 or tuple(logits.shape) != (t, observer.native.config.vocab_size):
        raise RuntimeError('Original full-vocabulary head shape/dtype changed')
    endpoints = token_endpoints(logits, axes['answer_ids'], mapping['answer_offsets'])
    for endpoint, row in zip(endpoints, receiver_target_axis[:-1], strict=True):
        endpoint['receiver_target_axis'] = row
    write_json(output / 'original-token-endpoints.json', dict(tokens=endpoints,
        final_posttoken_query_has_observed_next_target=False,
        probability_scope='Observer probability of original tokens; not original-generator sampling probability or risk'))
    clean = _hook_counts(observer.native) == hooks
    residency = observer.resource_report()
    if not clean or not residency['all_parameters_meta'] or residency['active_module'] is not None:
        raise RuntimeError('Observation hook/residency cleanup failed')
    return dict(actual_counts=dict(counts), archive_qualified=True, all_layers_AV_qualified=True,
        layers=len(observer.layers), physical_heads_per_layer=[observer.head_layout(i)[0]
        for i in range(len(observer.layers))], query_rows=len(rows), original_target_rows=t,
        AV_policy=AV_POLICY, phase=phase, hook_cleanup=clean, resource_report=residency,
        new_detector_scores=0, auroc=None, target_annotation_reads=0,
        original_generator_internal_states=False, source_reading_Q_observed=False,
        posttoken_trajectory_observed=True, causal_necessity_tested=False)


def _load_inputs(capture_root, prepared, alignment_root, license_path):
    license_ = read_json(license_path)
    if file_hash(license_path) != LICENSE_SHA256:
        raise ValueError('The independently reviewed manual source license changed')
    licensed = [row for row in license_['cases'] if row['id'] == CASE and row['source_id'] == SOURCE]
    if len(licensed) != 1:
        raise ValueError('Prescribed source license is missing or ambiguous')
    licensed = licensed[0]
    inputs, manifest = prepared / 'inputs.jsonl', prepared / 'manifest.json'
    input_sha = file_hash(inputs)
    if (input_sha != license_['provenance']['current_prepared_inputs']['sha256']
            or file_hash(manifest) != license_['provenance']['current_prepared_manifest']['sha256']
            or read_json(manifest).get('inputs_sha256') != input_sha):
        raise ValueError('Prepared original input identity changed')
    with inputs.open(encoding='utf-8') as stream:
        line = next((raw for i, raw in enumerate(stream, 1)
                     if i == licensed['prepared_source_match']['line']), None)
    if line is None:
        raise ValueError('Fixed prepared input line is missing')
    sample = Sample(**json.loads(line))  # Only this prescribed original Q/S/R is decoded.
    if (sample.id != CASE or sample.source_id != SOURCE or sample.task != 'QA' or sample.split != 'test'
            or sample.source != licensed['source'] or text_hash(sample.source) != licensed['source_sha256']):
        raise ValueError('Prescribed natural source/input changed')
    alignment_seal = verify_seal(alignment_root, required=('plan.json', f'answer-{CASE}.json', 'summary.json'))
    plan, previous = read_json(alignment_root / 'plan.json'), read_json(alignment_root / f'answer-{CASE}.json')
    if (previous.get('status') != 'original_axes_matched' or previous.get('id') != CASE
            or previous.get('source_id') != SOURCE or previous.get('response_sha256') != text_hash(sample.response)
            or input_sha not in plan['input_bindings'].values()):
        raise ValueError('Completed original M2 input/response join changed')
    dependencies = plan['source']['source_sha256']
    base = Path(__file__).parent
    for key, path in [('experiments/relation_alignment.py', base / 'relation_alignment.py'),
                      ('myproject/inputs.py', base.parent / 'myproject' / 'inputs.py'),
                      ('experiments.research/relation_inventory.py', base / 'research' / 'relation_inventory.py')]:
        if dependencies.get(key) != file_hash(path):
            raise ValueError('Completed M2 encoder/alignment dependency changed')
    directory = capture_root / 'capture' / CASE
    bindings = {str(inputs): input_sha, str(manifest): file_hash(manifest),
                str(license_path): LICENSE_SHA256, str(alignment_root / 'seal.json'): file_hash(alignment_root / 'seal.json')}
    for name in ('record.json', 'arrays.npz'):
        path = directory / name
        digest = file_hash(path)
        if (digest != licensed['M1_current_QA900_mapping']['matching_case']['files_sha256'][name]
                or plan['input_bindings'].get(str(path)) != digest):
            raise ValueError('Original capture differs from source/M2 witnesses')
        bindings[str(path)] = digest
    record = read_json(directory / 'record.json')
    for name in ('id', 'source_id', 'task', 'split', 'generator'):
        if str(record[name]) != str(getattr(sample, name)):
            raise ValueError('Original capture and selected inference input differ')
    p, t, n = LENGTHS
    audit = record['audit']
    if (tuple(audit[k] for k in ('prompt_length', 'answer_tokens', 'input_tokens')) != LENGTHS
            or audit['layer'] != 15 or audit['stop_after_layer'] is not True
            or audit['full_model_forwards'] != 0 or tuple(audit['sites']) != ('residual', 'attention_write', 'mlp_write')
            or [world['blocked'] for world in audit['worlds']] != [False, True]
            or any(world['executed_layers'] != 16 or world['full_model_completed'] is not False
                   for world in audit['worlds'])):
        raise ValueError('Original early-stop native archive contract changed')
    reference = _reference_payload(directory / 'arrays.npz', p, t, n)
    for name, shape in [('nodes', (2, t + 1, 3, 4096)), ('query', (2, t + 1, 32, 128)),
                        ('key', (2, n, 8, 128)), ('value', (2, n, 8, 128))]:
        if reference[name].dtype != np.float16 or reference[name].shape != shape:
            raise ValueError('Full original layer15 native coordinate shape/dtype changed')
    axes = {name: reference[name].tolist() for name in
            ('token_ids', 'answer_ids', 'source_mask', 'query_positions')}
    return sample, axes, reference, previous, bindings, alignment_seal


def run(args):
    """Central claimed stage, required Namespace fields:

    capture_root, prepared, model, asset_manifest, head_qualification,
    alignment_root, source_license, admission, output. Optional:
    asset_verification=None, asset_state=None, device='cuda:0'. No auto retry.
    """
    import torch
    from transformers import AutoTokenizer
    started = time.perf_counter()
    capture, prepared, model, head_root, aligned_root, license_path, admission, output = (
        _path(getattr(args, name)) for name in ('capture_root', 'prepared', 'model',
        'head_qualification', 'alignment_root', 'source_license', 'admission', 'output'))
    if output.exists():
        raise FileExistsError('Existing natural observation is retained, never automatically restarted')
    if any(output == root or root in output.parents for root in
           (capture, prepared, model, head_root, aligned_root)):
        raise ValueError('Observation output must be disjoint from immutable inputs')
    if file_hash(admission) != ADMISSION_SHA256:
        raise ValueError('Reviewed natural observation admission changed')
    assets = verify_assets(args.asset_manifest, model,
        verification=getattr(args, 'asset_verification', None), state=getattr(args, 'asset_state', None))
    observer_identity = capture_identity(capture, model / 'config.json')
    prerequisite = head_prerequisite(head_root, assets, observer_identity)
    identity = research_identity()
    output.mkdir(parents=True, exist_ok=False)
    counts, observer, bindings, alignment_seal = {'model_calls': 0, 'head_calls': 0}, None, {}, None
    device = torch.device(getattr(args, 'device', 'cuda:0'))
    try:
        sample, axes, reference, previous, bindings, alignment_seal = _load_inputs(
            capture, prepared, aligned_root, license_path)
        alignment_plan = read_json(aligned_root / 'plan.json')
        if (alignment_plan.get('observer') != observer_identity or any(
                assets['files'].get(row['name'], {}).get('sha256') != row['sha256']
                for row in alignment_plan['tokenizer_assets'])):
            raise ValueError('M2 observer/tokenizer assets differ from the original qualified observer')
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, local_files_only=True,
                                                  trust_remote_code=False)
        mapping = prepare_axes(sample, tokenizer, axes, previous)
        functions = frozen_capture_functions(capture)
        limits = ResourceLimits(max_sequence_tokens=512)
        write_json(output / 'plan.json', dict(schema='h017-natural-temporal-observation-plan-v1',
            id=CASE, source_id=SOURCE, source_runtime=identity, observer_capture_identity=observer_identity,
            admission_sha256=ADMISSION_SHA256, assets=assets, head_prerequisite=prerequisite,
            input_bindings=bindings, alignment_seal=alignment_seal, manual_source_license_sha256=LICENSE_SHA256,
            original_axes=axes, mapping=mapping, planned_counts={'model_calls': 1, 'head_calls': 1},
            full_input_tokens=441, prompt_tokens=365, answer_tokens=76, query_rows=77,
            physical_layers=list(range(32)), physical_heads=32, kv_heads=8,
            resource_limits=limits.__dict__, coordinate_policy=COORDINATE_POLICY, AV_policy=AV_POLICY,
            probability_scope='Full-model observer original-token probability, never risk',
            semantic_scope='Manual source license; future BC not causally available to earlier numeric KV, but predictable from priors',
            labels_read=False, new_detector_scores=0, automatic_retry=False, before=_resources(device)))
        store = SafeTensorStore(model, index_sha256=assets['files']['model.safetensors.index.json']['sha256'],
                                expected_files=assets['files'])
        observer = StreamedObserver(store, read_json(model / 'config.json'),
            config_sha256=assets['files']['config.json']['sha256'], device=device, limits=limits)
        if len(observer.layers) != 32 or any(observer.head_layout(i) != (32, 8, 128) for i in range(32)):
            raise ValueError('Natural original observer layer/head population changed')
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        result = observe_original(observer, axes, reference, mapping,
            functions['capture_selected_layer'], output, counts=counts)
        if result['actual_counts'] != {'model_calls': 1, 'head_calls': 1}:
            raise RuntimeError('Finite one-forward/one-head budget changed')
        result.update(schema='h017-natural-temporal-observation-result-v1', id=CASE, source_id=SOURCE,
            after=_resources(device), elapsed_seconds=time.perf_counter() - started,
            manual_source_response_temporal_conflict_preidentified=True,
            direct_text_verifier_baseline_measured=False,
            internal_detection_localization_cost_advantage='not_tested', automatic_retry=False)
        write_json(output / 'result.json', result)
    except BaseException as error:
        write_json(output / 'failure.json', dict(type=type(error).__name__, message=str(error),
            actual_counts=dict(counts), automatic_retry=False, elapsed_seconds=time.perf_counter() - started,
            resources=_resources(device if torch.cuda.is_initialized() else torch.device('cpu'))))
        raise
    finally:
        if observer is not None:
            observer.close()
    try:
        completed_identity(identity)
        if (file_hash(admission) != ADMISSION_SHA256
                or any(file_hash(Path(path)) != digest for path, digest in bindings.items())
                or verify_seal(aligned_root, expected=alignment_seal) != alignment_seal
                or verify_seal(head_root, expected=prerequisite['members']) != prerequisite['members']
                or verify_assets(args.asset_manifest, model, verification=getattr(args, 'asset_verification', None),
                                 state=getattr(args, 'asset_state', None)) != assets):
            raise RuntimeError('Frozen natural-observation dependency changed')
    except BaseException as error:
        write_json(output / 'failure.json', dict(type=type(error).__name__, message=str(error),
            stage='post-observation-dependency-check', actual_counts=dict(counts), automatic_retry=False,
            elapsed_seconds=time.perf_counter()-started, resources=_resources(device)))
        raise
    _seal(output)
    return result
