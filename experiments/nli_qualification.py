"""Frozen external-supervision NLI diagnostic; source chunks versus exact units.

This standard max-evidence baseline omits original answer/QA context and has
limited coreference handling. It is neither Qwen's matched control nor a new
causal mechanism or adopted target-unsupervised main method. No gold is read.
"""
from dataclasses import asdict
from copy import deepcopy
from importlib.metadata import version
import math
import os
from pathlib import Path
import re
import time

from . import data
from .datasets import read_samples
from .provenance import completed_identity, research_identity
from .research import focused_verdict as focus
from .semantic_qualification import ANNOTATION_NAMES, _available_ram, _digest, _model_identity, _seal

REPO = 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli'
REVISION = '6f5cf0a2b59cabb106aca4c287eed12e357e90eb'
HF_RUNTIME = {'torch': '2.8.0+cu128', 'transformers': '4.57.1'}
FILES = {'config.json', 'tokenizer_config.json', 'special_tokens_map.json',
         'added_tokens.json', 'tokenizer.json', 'spm.model', 'model.safetensors'}
PINNED_SHA256 = {
    'config.json': 'a6c616d6dabeacf90fd0e776c741d3f0f30a05533ccf0bd3b5b62e94cfaa8d57',
    'tokenizer_config.json': '557b3d33d3f41b81ad769244e506549e98a1857d41dd58160aacd4d98d710b5a',
    'special_tokens_map.json': '9463f61e1b109a8eb4688b829260d7c6b1e6dff04c98ff7269bb89e2b92369b9',
    'added_tokens.json': 'dc046d04c9b0ada7ae6f1dc89c465801799acdf0c9a6aab8c15a1b2d5ca4e91f',
    'tokenizer.json': '05402ffae6dd382a8491b1d29bfc139bec5d332662e86a026f433ce54c25c202',
    'spm.model': 'c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd',
    'model.safetensors': '06d6fd89edd4f97816831626daafbdb0b029cf63bae8edc0bccab1d64e2e7707'}
HEAD_LABELS = {'0': 'entailment', '1': 'neutral', '2': 'contradiction'}
DEFAULT_PROTOCOL = dict(schema='nli-qualification-v1', purpose='observations_only',
    expected_original_cases=34, expected_units=104, original_inputs_sha256=None,
    partition_policy='response_units_v1', max_pair_tokens=512, batch_size=8,
    overlap_rule='min(64,budget//4)', risk_definition='1-minus-max-chunk-entailment',
    class_order=['supported', 'contradicted', 'unresolved'], unknown_risk=1.0,
    context_policy='source-premise-exact-unit-hypothesis-no-answer-or-QA-context',
    attention_implementation='eager', dtype='float16', token_projection_performed=False)


def _identity(value):
    value = _model_identity(value)
    if (value['model_id'] != REPO or value.get('revision') != REVISION
            or value['model_files_sha256'] != PINNED_SHA256 or value.get('runtime') != HF_RUNTIME):
        raise ValueError('Frozen seven-file DeBERTa revision/runtime identity required')
    return value


def _protocol(path):
    value = data.read_json(path)
    if (not isinstance(value, dict) or set(value) - set(DEFAULT_PROTOCOL) - {'notes', 'hypothesis_id'}
            or any(type(value.get(k)) is not type(v) or value[k] != v
                   for k, v in DEFAULT_PROTOCOL.items() if k != 'original_inputs_sha256')
            or not isinstance(value.get('original_inputs_sha256'), str)
            or re.fullmatch('[0-9a-f]{64}', value['original_inputs_sha256']) is None):
        raise ValueError('Prespecified full-population NLI protocol required')
    return value


def _prepare_pairs(tokenizer, source, hypothesis):
    """Tokenize without truncation; window every original source token ID."""
    source_ids = tokenizer.encode(source, add_special_tokens=False, truncation=False)
    hypothesis_ids = tokenizer.encode(hypothesis, add_special_tokens=False, truncation=False)
    if tokenizer.num_special_tokens_to_add(pair=True) != 3:
        raise ValueError('NLI pair framing must contain exactly three special tokens')
    budget = 512 - 3 - len(hypothesis_ids)
    reason = ('hypothesis_exhausts_pair_budget' if budget <= 0 else
              'empty_tokenized_source' if not source_ids else None)
    overlap = min(64, max(0, budget // 4))
    result = dict(source_tokens=len(source_ids), hypothesis_tokens=len(hypothesis_ids),
        budget=budget, overlap_tokens=overlap, status='unknown' if reason else 'scored',
        reason=reason, chunks=[])
    if reason:
        return result
    start = 0
    while start < len(source_ids):
        end = min(start + budget, len(source_ids))
        ids = tokenizer.build_inputs_with_special_tokens(source_ids[start:end], hypothesis_ids)
        if len(ids) != end - start + len(hypothesis_ids) + 3 or len(ids) > 512:
            raise ValueError('NLI pair framing changed or silently truncated tokens')
        result['chunks'].append(dict(start=start, end=end, input_ids=ids, attention_mask=[1] * len(ids)))
        if end == len(source_ids):
            break
        start = end - overlap
    return result


def _valid_pair(pair):
    ids, mask = pair.get('input_ids'), pair.get('attention_mask')
    if (not isinstance(ids, list) or not 3 < len(ids) <= 512
            or any(type(value) is not int or not 0 <= value < 128100 for value in ids)
            or not isinstance(mask, list) or len(mask) != len(ids)
            or any(type(value) is not int or value != 1 for value in mask)):
        raise ValueError('Invalid unpadded NLI token pair')


def _scores(logits):
    peak = max(logits)
    values = [math.exp(value - peak) for value in logits]
    total = math.fsum(values)
    return [values[index] / total for index in (0, 2, 1)]


def _prepare(samples, backend):
    if len(samples) != 34:
        raise ValueError('The full original 34-case population is required')
    records, pairs = [], {}
    for sample in samples:
        if not all(isinstance(value, str) for value in (sample.source, sample.response, sample.prompt)):
            raise ValueError('NLI source/response/prompt must be original strings')
        row = dict(asdict(sample), input_text_sha256={key: data.text_hash(value) for key, value in
                   dict(prompt=sample.prompt, source=sample.source, response=sample.response).items()}, units=[])
        for index, span in enumerate(focus.response_units(sample.response)):
            prepared = backend.prepare_pairs(sample.source, span['text'])
            unit = dict(span, index=index, **{k: prepared[k] for k in
                ('source_tokens', 'hypothesis_tokens', 'budget', 'overlap_tokens', 'status', 'reason')}, chunks=[])
            if (prepared['budget'] != 509 - prepared['hypothesis_tokens']
                    or prepared['overlap_tokens'] != min(64, max(0, prepared['budget'] // 4))):
                raise ValueError('Backend changed the frozen source-window budget')
            expected_reason = ('hypothesis_exhausts_pair_budget' if prepared['budget'] <= 0 else
                               'empty_tokenized_source' if prepared['source_tokens'] == 0 else None)
            if (prepared['reason'] != expected_reason or prepared['status'] != ('unknown' if expected_reason else 'scored')
                    or bool(prepared['chunks']) != (expected_reason is None)):
                raise ValueError('Backend unknown/chunk status changed')
            cursor = 0
            for chunk in prepared['chunks']:
                pair = {key: chunk[key] for key in ('input_ids', 'attention_mask')}
                _valid_pair(pair)
                end = min(cursor + prepared['budget'], prepared['source_tokens'])
                if (not cursor < end or chunk['start'] != cursor or chunk['end'] != end
                        or len(pair['input_ids']) != end - cursor + prepared['hypothesis_tokens'] + 3):
                    raise ValueError('Source chunks omitted tokens or changed pair length')
                key = _digest(pair)
                pairs[key] = pair
                unit['chunks'].append(dict(start=cursor, end=end, key=key))
                cursor = end - prepared['overlap_tokens'] if end < prepared['source_tokens'] else end
            if prepared['chunks'] and unit['chunks'][-1]['end'] != prepared['source_tokens']:
                raise ValueError('Source chunks do not cover all source tokens')
            row['units'].append(unit)
        records.append(row)
    if sum(len(row['units']) for row in records) != 104:
        raise ValueError('The same complete 104-unit focus partition is required')
    batches = [list(pairs)[start:start + 8] for start in range(0, len(pairs), 8)]
    return records, pairs, batches


def _plan(inputs, input_hash, protocol_hash, frozen, identity, source, records, pairs, batches):
    return dict(schema='nli-qualification-plan-v1', state='frozen_before_observation',
        original_inputs_path=str(inputs), original_inputs_sha256=input_hash, protocol_sha256=protocol_hash,
        protocol=frozen, model_identity=identity, source_identity=source, records=records, pairs=pairs, batches=batches,
        head_id2label=HEAD_LABELS, semantic_head_indices=[0, 2, 1],
        donor=None, evaluation_material_accessed=False, token_projection_performed=False,
        limitations=['External NLI supervision; no target-label fitting or contamination-absence claim.',
                    'Max-evidence diagnostic, not causal or novel; omits original answer/QA context and coreference resolution.'])


def _costs(records, pairs):
    costs = dict(logical_pairs=sum(len(u['chunks']) for r in records for u in r['units']),
        unique_pairs=len(pairs), exact_pair_reuses=0, actual_forward_passes=0, input_tokens=0,
        padded_input_tokens=0, seconds=0.0)
    costs['exact_pair_reuses'] = costs['logical_pairs'] - costs['unique_pairs']
    return costs


def _batch(raw, keys, pairs, index):
    count = sum(len(pairs[key]['input_ids']) for key in keys)
    padded = len(keys) * max(len(pairs[key]['input_ids']) for key in keys)
    logits = raw.get('logits') if isinstance(raw, dict) else None
    if (not isinstance(logits, list) or len(logits) != len(keys)
            or any(not isinstance(row, list) or len(row) != 3 or any(type(x) not in (int, float) or
                   not math.isfinite(x) for x in row) for row in logits)
            or raw.get('actual_forward_passes') != 1 or type(raw.get('actual_forward_passes')) is not int
            or any(type(raw.get(name)) is not int for name in ('input_tokens', 'padded_input_tokens'))
            or raw.get('input_tokens') != count or raw.get('padded_input_tokens') != padded
            or type(raw.get('seconds')) not in (int, float) or not math.isfinite(raw['seconds']) or raw['seconds'] < 0):
        raise ValueError('Invalid batched three-head NLI logits/costs')
    return ({key: dict(logits=values, class_vector=_scores(values), batch=index)
             for key, values in zip(keys, logits)},
            {name: raw[name] for name in ('actual_forward_passes', 'input_tokens', 'padded_input_tokens', 'seconds')})


def _completion(records, observations, frozen, source, identity, costs):
    for row in records:
        for unit in row['units']:
            for chunk in unit['chunks']:
                chunk.update(observations[chunk['key']])
            selected = max(unit['chunks'], key=lambda c: c['class_vector'][0]) if unit['chunks'] else None
            vector = selected['class_vector'] if selected else None
            unit.update(selected_chunk=selected, class_vector=vector,
                class_scores=dict(zip(frozen['class_order'], vector)) if vector else None,
                risk=1.0 - vector[0] if vector else 1.0, unknown_fallback=selected is None)
    result = dict(schema='nli-qualification-complete-v1', state='observations_sealed_not_evaluated',
        records=records, costs=costs, coverage=dict(original_cases=len(records), units=104,
        scored_units=sum(u['status'] == 'scored' for r in records for u in r['units']),
        unknown_units=sum(u['unknown_fallback'] for r in records for u in r['units'])),
        donor=None, evaluation_material_accessed=False, token_projection_performed=False, auroc_computed=False,
        source_identity_sha256=_digest(source), model_identity_sha256=_digest(identity))
    return result


def qualify(inputs, protocol, output, backend, model_identity):
    """Freeze all originals/units/pairs before batched three-head observations."""
    inputs, protocol, output = (Path(p).absolute() for p in (inputs, protocol, output))
    if (any(path.resolve() != path for path in (inputs, protocol, output))
            or inputs.name != 'inputs.jsonl' or protocol.name.casefold() in ANNOTATION_NAMES
            or 'outputs' not in output.parts or inputs.is_relative_to(output) or protocol.is_relative_to(output)):
        raise ValueError('Original inputs and independent protocol/retained outputs required')
    if output.exists():
        raise FileExistsError('Existing NLI output requires inspection; automatic retry is forbidden')
    frozen, identity = _protocol(protocol), _identity(model_identity)
    input_hash, protocol_hash = data.file_hash(inputs), data.file_hash(protocol)
    if input_hash != frozen['original_inputs_sha256']:
        raise ValueError('Original input hash differs from frozen protocol')
    if getattr(backend, 'model_identity', None) != identity:
        raise ValueError('Backend model identity differs')
    samples, source = list(read_samples(inputs)), research_identity()
    records, pairs, batches = _prepare(samples, backend)
    plan = _plan(inputs, input_hash, protocol_hash, frozen, identity, source, records, pairs, batches)
    output.mkdir(parents=True)
    for name, value in [('plan.json', plan), ('protocol.json', frozen), ('model_identity.json', identity),
                        ('source_identity.json', source)]:
        data.write_json(output / name, value)
    for index, keys in enumerate(batches):
        root = output / 'batches' / f'{index:05d}'
        root.mkdir(parents=True)
        data.write_json(root / 'request.json', dict(keys=keys, pairs=[pairs[key] for key in keys]))
    immutable = {str(path.relative_to(output)): data.file_hash(path) for path in output.rglob('*') if path.is_file()}
    data.write_json(output / 'freeze.json', immutable)
    freeze_hash = data.file_hash(output / 'freeze.json')
    def check_frozen():
        if (data.file_hash(inputs) != input_hash or data.file_hash(protocol) != protocol_hash
                or data.file_hash(output / 'freeze.json') != freeze_hash
                or getattr(backend, 'model_identity', None) != identity
                or any(data.file_hash(output / name) != digest for name, digest in immutable.items())):
            raise ValueError('Frozen NLI population/protocol/identity changed')
    costs = _costs(records, pairs)
    observations = {}
    for index, keys in enumerate(batches):
        check_frozen()
        root = output / 'batches' / f'{index:05d}'
        count = sum(len(pairs[key]['input_ids']) for key in keys)
        padded = len(keys) * max(len(pairs[key]['input_ids']) for key in keys)
        request = deepcopy([pairs[key] for key in keys])
        raw = backend.classify(request)
        data.write_json(root / 'backend_raw.json', {'value': raw})
        if len(request) != len(keys) or any(_digest(pair) != key for pair, key in zip(request, keys)):
            raise ValueError('Backend mutated a frozen NLI token pair')
        batch_values, batch_costs = _batch(raw, keys, pairs, index)
        observations.update(batch_values)
        for name, value in batch_costs.items():
            costs[name] += value
        _seal(root, ['request.json', 'backend_raw.json'])
    check_frozen()
    completed_identity(source)
    result = _completion(records, observations, frozen, source, identity, costs)
    data.write_json(output / 'complete.json', result)
    _seal(output, [str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()])
    data.verify_seal(output)
    return result


class HFEntailment:
    """Tokenizer-only planning; explicitly admitted weights load at classify()."""
    def __init__(self, model, identity, admit_resources=False, execution_mode='legacy-pilot'):
        if execution_mode not in {'legacy-pilot','dataset-fp32-batch1-v1'}:
            raise ValueError('An explicit frozen NLI execution mode is required')
        self.model_path, self.model_identity = Path(model).resolve(), _identity(identity)
        self.admitted, self.tokenizer, self.model = admit_resources, None, None
        self.execution_mode, self._precision_restore = execution_mode, None

    def verify_frozen_identity(self):
        files = self.model_identity['model_files_sha256']
        if {p.name for p in self.model_path.iterdir() if p.is_file()} != FILES:
            raise ValueError('Frozen NLI model directory is incomplete or has extra files')
        if any((self.model_path / name).is_symlink() or data.file_hash(self.model_path / name) != digest
               for name, digest in files.items()):
            raise ValueError('Frozen NLI model asset changed')
        config = data.read_json(self.model_path / 'config.json')
        if (config.get('id2label') != HEAD_LABELS or config.get('max_position_embeddings') != 512
                or config.get('architectures') != ['DebertaV2ForSequenceClassification']):
            raise ValueError('Frozen NLI classifier head/architecture/length differs')

    def prepare_pairs(self, source, hypothesis):
        if self.tokenizer is None:
            self.verify_frozen_identity()
            if version('transformers') != HF_RUNTIME['transformers']:
                raise RuntimeError('Pinned Transformers required for tokenizer-only planning')
            os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True,
                trust_remote_code=False, use_fast=True)
            if not self.tokenizer.is_fast:
                raise ValueError('Saved fast tokenizer required; do not install SentencePiece')
        return _prepare_pairs(self.tokenizer, source, hypothesis)

    def classify(self, pairs):
        if not self.admitted:
            raise RuntimeError('Explicit NLI resource admission required before weights/forwards')
        dataset_mode = self.execution_mode == 'dataset-fp32-batch1-v1'
        if not 1 <= len(pairs) <= (1 if dataset_mode else 8):
            raise ValueError('Dataset NLI requires batch1' if dataset_mode else
                             'Frozen NLI batches contain one through eight pairs')
        for pair in pairs:
            _valid_pair(pair)
        import torch
        from transformers import AutoModelForSequenceClassification
        floor = 512 * 1024**2
        if dataset_mode and (_available_ram() < floor
                or not torch.cuda.is_available() or torch.cuda.mem_get_info(0)[0] < floor):
            raise RuntimeError('Dataset NLI requires the frozen512MiB host/GPU floors')
        if self.model is None:
            if self.tokenizer is None:
                raise RuntimeError('Freeze tokenizer-only pair planning before classification')
            self.verify_frozen_identity()
            if (any(version(name) != pin for name, pin in HF_RUNTIME.items())
                    or _available_ram() < (floor if dataset_mode else 2 * 1024**3)
                    or not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0)
                    or torch.cuda.mem_get_info(0)[0] < 5 * 1024**3 // 2):
                raise RuntimeError('Pinned SM120 / 512MiB RAM / 2.5GiB free GPU required' if dataset_mode else
                                   'Pinned SM120 runtime / 2 GiB RAM / 2.5 GiB free GPU required')
            free, total = torch.cuda.mem_get_info(0)
            cap = (6 if dataset_mode else 2) * 1024**3
            if dataset_mode:
                self._precision_restore = (torch.backends.cuda.matmul.allow_tf32,
                    torch.backends.cudnn.allow_tf32,torch.cuda.get_per_process_memory_fraction(0))
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            torch.cuda.set_per_process_memory_fraction(cap / total, 0)
            self.admission = dict(available_ram_bytes=_available_ram(), free_cuda_bytes=free,
                cuda_allocator_limit_bytes=cap, device='cuda:0')
            if dataset_mode:
                self.admission.update(execution_mode=self.execution_mode,host_floor_bytes=floor,
                    gpu_floor_bytes=floor,weight_storage='float16',arithmetic='float32',
                    loading='native safetensors meta device-map to CUDA; no full CPU FP32 copy')
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path,
                    local_files_only=True,trust_remote_code=False,use_safetensors=True,
                    torch_dtype=torch.float32,attn_implementation='eager',
                    device_map={'':'cuda:0'},low_cpu_mem_usage=True).eval().requires_grad_(False)
                if any(p.dtype!=torch.float32 or p.is_meta or p.device.type!='cuda'
                       for p in self.model.parameters()):
                    raise RuntimeError('Dataset NLI native FP32 CUDA placement changed')
            else:
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path,
                    local_files_only=True, trust_remote_code=False, use_safetensors=True,
                    torch_dtype=torch.float16, attn_implementation='eager').eval().to('cuda:0')
        if dataset_mode:
            from .streamed_qualification import _resources
            resources_before = _resources(torch.device('cuda:0'))
        encoded = self.tokenizer.pad(pairs, padding=True, return_tensors='pt').to('cuda:0')
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        started = time.monotonic()
        with torch.inference_mode():
            if dataset_mode and (torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('cuda')
                    or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
                    or self.model.training):
                raise RuntimeError('Dataset NLI inference precision context changed')
            logits = self.model(input_ids=encoded['input_ids'], attention_mask=encoded['attention_mask']).logits
        torch.cuda.synchronize(0)
        if (tuple(logits.shape) != (len(pairs), 3) or not bool(torch.isfinite(logits).all().item())
                or (dataset_mode and logits.dtype!=torch.float32)):
            raise ValueError('NLI model produced invalid batch shape/finiteness')
        result = dict(logits=logits.double().cpu().tolist(), actual_forward_passes=1,
            input_tokens=sum(len(pair['input_ids']) for pair in pairs), padded_input_tokens=encoded['input_ids'].numel(),
            seconds=time.monotonic() - started, peak_cuda_bytes=torch.cuda.max_memory_reserved(0),
            resource_admission=self.admission)
        if dataset_mode:
            result.update(resources_before=resources_before,resources_after=_resources(torch.device('cuda:0')))
        return result

    def close(self):
        """Release this backend and restore the optional dataset-mode settings."""
        self.model = None
        self.admitted = False
        if self._precision_restore is not None:
            import torch
            matmul,cudnn,fraction = self._precision_restore
            torch.backends.cuda.matmul.allow_tf32 = matmul
            torch.backends.cudnn.allow_tf32 = cudnn
            torch.cuda.set_per_process_memory_fraction(fraction,0)
            torch.cuda.empty_cache()
            self._precision_restore = None
