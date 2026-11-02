"""Seal source observations for a frozen exploratory panel, without evaluating truth.

The callable generator receives messages and a fixed generation limit, never sample
IDs or panel annotations. Heavy HF imports and weight loading are explicitly gated.
"""

from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import time

from . import data
from .datasets import read_samples
from .provenance import completed_identity, research_identity
from .research import semantic_probe as probe


DEFAULT_PROTOCOL = {
    'schema': 'semantic-qualification-v2', 'contract_version': probe.CONTRACT_VERSION,
    'context_policy': probe.CONTEXT_POLICY,
    'purpose': 'observations_only', 'max_input_tokens': 4096,
    'proposer_max_new_tokens': 1536, 'observation_max_new_tokens': 256,
    'do_sample': False, 'num_beams': 1, 'attention_implementation': 'sdpa',
}
MODES = ('direct', 'blind', 'leaky')
HF_RUNTIME = {'torch': '2.8.0+cu128', 'transformers': '4.57.1',
              'accelerate': '1.10.1', 'bitsandbytes': '0.48.0'}
ANNOTATION_NAMES = {'evaluation_only.jsonl', 'annotations.jsonl'}


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      allow_nan=False)


def _digest(value):
    return data.text_hash(_canonical(value))


def _protocol(path):
    value = data.read_json(path)
    if not isinstance(value, dict) or any(value.get(k) != v or type(value.get(k)) is not type(v)
                                         for k, v in DEFAULT_PROTOCOL.items()):
        raise ValueError('Unsupported observation protocol: preserve the fixed v2 context policy and observation/decode limits')
    if set(value) - set(DEFAULT_PROTOCOL) - {'notes', 'hypothesis_id'}:
        raise ValueError('Unsupported protocol field')
    return value


def _model_identity(value):
    # Deep-copy JSON metadata; this function never follows its model-file paths.
    value = json.loads(_canonical(value))
    if (not isinstance(value, dict) or not isinstance(value.get('model_id'), str)
            or not value['model_id'] or not isinstance(value.get('model_files_sha256'), dict)
            or not value['model_files_sha256']):
        raise ValueError('A frozen model ID and nonempty model-file SHA256 map are required')
    for name, digest in value['model_files_sha256'].items():
        if (not isinstance(name, str) or not name or not isinstance(digest, str)
                or re.fullmatch('[0-9a-f]{64}', digest) is None):
            raise ValueError('Invalid frozen model-file identity')
    return value


def _seal(root, names):
    data.write_json(root / 'seal.json', {name: data.file_hash(root / name) for name in names})


def _request(generator, request, protocol, identity, requests):
    limit = (protocol['proposer_max_new_tokens'] if request['mode'] == 'proposal'
             else protocol['observation_max_new_tokens'])
    execution = {'messages': request['messages'], 'model_identity': identity,
                 'decode': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': limit},
                 'admission': {'max_input_tokens': protocol['max_input_tokens']}}
    key = _digest(execution)
    root = _artifact_path(requests, key)
    if root.exists():
        if not (root / 'seal.json').is_file():
            raise RuntimeError('Unfinished request cache requires explicit artifact recovery: ' + key)
        data.verify_seal(root, expected=_request_members(root))
        if data.read_json(root / 'request.json') != execution:
            raise ValueError('Cached request identity changed')
        return key, data.read_json(root / 'response.json'), True
    root.mkdir()
    data.write_json(root / 'request.json', execution)
    count = generator.count_input_tokens(request['messages'])
    if type(count) is not int or count < 0:
        raise ValueError('Generator token counter must return a nonnegative integer')
    if count > protocol['max_input_tokens']:
        result = {'status': 'input_rejected', 'input_tokens': count, 'output_tokens': 0,
                  'text': None, 'reason': 'input_token_limit', 'actual_model_calls': 0,
                  'seconds': 0.0}
    else:
        started = time.monotonic()
        raw = generator(request['messages'], max_new_tokens=limit)
        # Persist the complete backend return before validating it. A bad backend
        # response remains an explicit incomplete cache, never a guessed observation.
        data.write_json(root / 'backend_raw.json', {'value': raw})
        if (not isinstance(raw, dict) or not isinstance(raw.get('text'), str)
                or type(raw.get('input_tokens')) is not int or raw['input_tokens'] != count
                or type(raw.get('output_tokens')) is not int
                or not 0 <= raw['output_tokens'] <= limit
                or not isinstance(raw.get('finish_reason'), str)):
            raise ValueError('Generator returned invalid text/token-count/finish metadata')
        result = {**raw, 'status': 'generated', 'actual_model_calls': 1,
                  'seconds': time.monotonic() - started}
    data.write_json(root / 'response.json', result)
    names = ['request.json', 'response.json']
    if (root / 'backend_raw.json').is_file():
        names.append('backend_raw.json')
    _seal(root, names)
    return key, result, False


def _execute(generator, request, protocol, identity, requests):
    key, result, reused = _request(generator, request, protocol, identity, requests)
    return {'mode': request['mode'], 'request_key': key, 'cache_reused': reused,
            'execution': result}


def _context(inputs):
    context = probe.pre_generation_context(inputs['prompt'], inputs['source'])
    start = inputs['prompt'].index(inputs['source'])
    transformation = {'policy': probe.CONTEXT_POLICY, 'placeholder': probe.SOURCE_PLACEHOLDER,
                      'original_source_span': [start, start + len(inputs['source'])],
                      'other_prompt_characters_preserved': True}
    return context, transformation


def _case(sample, generator, protocol, identity, requests, context_info):
    inputs = sample.inputs
    context, transformation = context_info
    proposal = _execute(generator, probe.build_proposer(inputs['response'], context=context),
                        protocol, identity, requests)
    if proposal['execution']['status'] == 'input_rejected':
        parsed = {'frames': (), 'invalid': (), 'raw_claim_count': None,
                  'envelope_error': {'code': 'input_token_limit', 'message': 'Proposal input rejected'}}
    else:
        parsed = probe.parse_proposal_items(proposal['execution']['text'], inputs['response'])
    frames = parsed['frames']
    output_limited = proposal['execution'].get('finish_reason') == 'max_new_tokens'
    limited_reasons = ([proposal['execution']['reason']] if proposal['execution']['status'] == 'input_rejected' else [])
    if output_limited:
        limited_reasons.append('proposal_output_token_limit')
    if parsed['envelope_error'] is not None:
        limited_reasons.append('proposal_envelope_invalid')
    if parsed['invalid']:
        limited_reasons.append('invalid_proposed_frames')
    if any(not frame.slots for frame in frames):
        limited_reasons.append('frames_without_slots')
    observations = []
    for frame_index, frame in enumerate(frames):
        for slot_index, slot in enumerate(frame.slots):
            for mode in MODES:
                item = {'frame_index': frame_index, 'slot_index': slot_index, 'mode': mode}
                try:
                    request = getattr(probe, 'build_' + mode)(inputs['source'], inputs['response'], frame, slot,
                                                             context=context)
                except probe.ProbeValidationError as error:
                    item.update(parse_status='request_invalid', error={'code': error.code, 'message': str(error)},
                                observation=asdict(probe.unknown_observation('request_invalid:' + error.code)))
                else:
                    item.update(_execute(generator, request, protocol, identity, requests))
                    raw = item['execution']
                    if raw['status'] == 'input_rejected':
                        item.update(parse_status='input_rejected', observation=asdict(
                            probe.unknown_observation('input_token_limit')))
                    else:
                        try:
                            observation = probe.parse_observation(raw['text'], inputs['source'], mode=mode)
                        except probe.ProbeValidationError as error:
                            item.update(parse_status='invalid', error={'code': error.code, 'message': str(error)},
                                        observation=asdict(probe.unknown_observation('invalid_output:' + error.code)))
                        else:
                            item.update(parse_status='valid', observation=asdict(observation))
                observations.append(item)
    return {'schema': 'semantic-qualification-case-v2', 'id': sample.id,
            'source_id': sample.source_id, 'task': sample.task, 'generator': sample.generator,
            'input_text_sha256': {name: data.text_hash(text) for name, text in inputs.items()},
            'pre_generation_context_sha256': data.text_hash(context),
            'context_transformation': transformation,
            'proposal': proposal, 'raw_claim_count': parsed['raw_claim_count'],
            'proposal_envelope_error': parsed['envelope_error'],
            'invalid_frames': parsed['invalid'], 'frames': [asdict(frame) for frame in frames],
            'unobserved_frames': [{'frame_index': i, 'status': 'unknown', 'reason': 'no_proposed_slots'}
                                  for i, frame in enumerate(frames) if not frame.slots],
            'coverage': probe.proposal_coverage(inputs['response'], frames),
            'proposal_output_limited': output_limited, 'coverage_limited_reasons': limited_reasons,
            'observations': observations, 'state': 'observations_sealed_not_evaluated'}


def _request_members(root):
    names = {'request.json', 'response.json'}
    if (root / 'backend_raw.json').is_file():
        names.add('backend_raw.json')
    if {p.name for p in root.iterdir()} != names | {'seal.json'}:
        raise ValueError('Unexpected or incomplete request-cache members')
    return names


def _artifact_path(parent, name):
    path = parent / name
    if path.resolve() != path:
        raise ValueError('Redirected qualification artifact directory')
    return path


def qualify(inputs, protocol, output, generator, model_identity):
    """Collect observations only; generator has count_input_tokens and __call__.

    A completed case can resume only under the identical input/protocol/code/model
    plan. Unfinished cases/cache/locks require explicit recovery, never a rerun.
    """
    inputs, protocol, output = Path(inputs).resolve(), Path(protocol).resolve(), Path(output).resolve()
    if inputs.name != 'inputs.jsonl' or protocol.name.casefold() in ANNOTATION_NAMES:
        raise ValueError('Only the original inputs.jsonl and an independent protocol may be read')
    if 'outputs' not in output.parts:
        raise ValueError('Qualification artifacts must be retained under outputs/')
    frozen_protocol, identity, source = _protocol(protocol), _model_identity(model_identity), research_identity()
    if hasattr(generator, 'model_identity') and generator.model_identity != identity:
        raise ValueError('Generator and supplied model identities differ')
    input_hash, protocol_hash = data.file_hash(inputs), data.file_hash(protocol)
    samples = list(read_samples(inputs))
    if not samples:
        raise ValueError('Qualification requires nonempty original inference inputs')
    contexts = {s.id: _context(s.inputs) for s in samples}
    roster = [{'id': s.id, 'source_id': s.source_id, 'response_sha256': data.text_hash(s.inputs['response'])}
              for s in samples]
    if data.file_hash(inputs) != input_hash or data.file_hash(protocol) != protocol_hash:
        raise ValueError('Inputs or protocol changed while being frozen')
    plan = {'schema': 'semantic-qualification-plan-v2', 'state': 'observations_only',
            'inputs_path': str(inputs), 'inputs_sha256': input_hash, 'protocol_sha256': protocol_hash,
            'protocol': frozen_protocol, 'model_identity_sha256': _digest(identity),
            'source_identity_sha256': _digest(source), 'records': roster,
            'evaluation_material_accessed': False, 'case_ids_are_model_inputs': False,
            'engineering_failure_is_semantic_refutation': False}
    output.mkdir(parents=True, exist_ok=True)
    lock = output / '.qualification.lock'
    try:
        lock_descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError('Existing qualification claim requires explicit liveness/recovery inspection') from error
    os.close(lock_descriptor)
    try:
        for name in ['plan.json', 'protocol.json', 'model_identity.json', 'source_identity.json',
                     'complete.json', 'seal.json']:
            _artifact_path(output, name)
        if (output / 'plan.json').exists():
            if data.read_json(output / 'plan.json') != plan:
                raise ValueError('Resume plan differs: inputs/protocol/model/code must remain identical')
            for name, expected in [('protocol.json', frozen_protocol), ('model_identity.json', identity),
                                   ('source_identity.json', source)]:
                if data.read_json(output / name) != expected:
                    raise ValueError('Frozen qualification identity changed: ' + name)
        else:
            if set(p.name for p in output.iterdir()) != {lock.name}:
                raise RuntimeError('Unfinished qualification metadata requires explicit recovery')
            data.write_json(output / 'protocol.json', frozen_protocol)
            data.write_json(output / 'model_identity.json', identity)
            data.write_json(output / 'source_identity.json', source)
            data.write_json(output / 'plan.json', plan)
        cases, requests = _artifact_path(output, 'cases'), _artifact_path(output, 'requests')
        cases.mkdir(exist_ok=True); requests.mkdir(exist_ok=True)
        allowed_cases = {data.text_hash(s.id) for s in samples}
        if set(p.name for p in cases.iterdir()) - allowed_cases:
            raise ValueError('Unexpected qualification case artifacts')
        if set(p.name for p in output.iterdir()) - {
                '.qualification.lock', 'plan.json', 'protocol.json', 'model_identity.json',
                'source_identity.json', 'complete.json', 'seal.json', 'cases', 'requests'}:
            raise ValueError('Unexpected qualification root artifacts')
        records, all_events, case_values = [], [], []
        for sample in samples:
            root = _artifact_path(cases, data.text_hash(sample.id))
            if root.exists():
                if not (root / 'seal.json').is_file():
                    raise RuntimeError('Unfinished case requires explicit artifact recovery: ' + sample.id)
                if {p.name for p in root.iterdir()} != {'case.json', 'seal.json'}:
                    raise ValueError('Unexpected case artifact members')
                data.verify_seal(root, expected={'case.json'})
                case = data.read_json(root / 'case.json')
                if (case['id'] != sample.id or case['input_text_sha256'] !=
                        {name: data.text_hash(text) for name, text in sample.inputs.items()}
                        or case.get('schema') != 'semantic-qualification-case-v2'
                        or case.get('pre_generation_context_sha256') != data.text_hash(contexts[sample.id][0])
                        or case.get('context_transformation') != contexts[sample.id][1]):
                    raise ValueError('Sealed case input identity changed')
            else:
                if (output / 'seal.json').exists():
                    raise ValueError('Completed qualification is missing a case')
                root.mkdir()
                case = _case(sample, generator, frozen_protocol, identity, requests, contexts[sample.id])
                data.write_json(root / 'case.json', case)
                _seal(root, ['case.json'])
            records.append({'id': sample.id, 'file': str((root / 'case.json').relative_to(output)),
                            'sha256': data.file_hash(root / 'case.json')})
            case_values.append(case)
            all_events.extend([case['proposal']] + case['observations'])
        keys = {e['request_key'] for e in all_events if 'request_key' in e}
        if set(p.name for p in requests.iterdir()) != keys:
            raise ValueError('Unreferenced or missing request cache requires explicit recovery')
        costs, cache_names = [], []
        for key in sorted(keys):
            root = _artifact_path(requests, key)
            members = _request_members(root)
            data.verify_seal(root, expected=members)
            execution = data.read_json(root / 'request.json')
            if (_digest(execution) != key or execution['model_identity'] != identity):
                raise ValueError('Request-cache identity mismatch')
            costs.append(data.read_json(root / 'response.json'))
            cache_names.extend(str((root / name).relative_to(output)) for name in sorted(members | {'seal.json'}))
        completed_identity(source)
        if data.file_hash(inputs) != input_hash or data.file_hash(protocol) != protocol_hash:
            raise ValueError('Original inputs/protocol changed during qualification')
        if hasattr(generator, 'verify_frozen_identity'):
            generator.verify_frozen_identity()
        if [record['id'] for record in records] != [sample.id for sample in samples]:
            raise ValueError('Qualification did not retain the entire original case roster')
        complete = {'schema': 'semantic-qualification-complete-v2',
                    'state': 'observations_sealed_not_evaluated', 'records': records,
                    'input_id_order_sha256': _digest([sample.id for sample in samples]),
                    'coverage': {'cases': len(records),
                        'raw_claims': sum(c['raw_claim_count'] or 0 for c in case_values),
                        'valid_frames': sum(c['coverage']['claims'] for c in case_values),
                        'valid_slots': sum(c['coverage']['slots'] for c in case_values),
                        'frames_without_slots': sum(c['coverage']['claims_without_slots'] for c in case_values),
                        'invalid_frames': sum(len(c['invalid_frames']) for c in case_values),
                        'proposal_envelope_failures': sum(c['proposal_envelope_error'] is not None for c in case_values),
                        'proposal_output_token_limits': sum(c['proposal_output_limited'] for c in case_values),
                        'cases_with_coverage_limitations': sum(bool(c['coverage_limited_reasons']) for c in case_values),
                        'cases_without_valid_frames': sum(not c['frames'] for c in case_values),
                        'semantic_exhaustiveness_verified': False,
                        'observation_statuses': dict(Counter(e.get('parse_status') for e in all_events
                                                             if e['mode'] != 'proposal')),
                        'observation_output_token_limits': sum(e.get('execution', {}).get('finish_reason') ==
                            'max_new_tokens' for e in all_events if e['mode'] != 'proposal'),
                        'unknown_observations': sum(e['observation']['status'] == 'unknown'
                                                   for e in all_events if e['mode'] != 'proposal')},
                    'costs': {'logical_requests': sum('request_key' in e for e in all_events),
                              'scope': 'retained artifact lifetime; completed resume adds no model calls',
                              'unique_requests': len(keys), 'actual_model_calls': sum(c['actual_model_calls'] for c in costs),
                              'exact_request_reuses': sum('request_key' in e for e in all_events) - len(keys),
                              'input_tokens_generated': sum(c['input_tokens'] for c in costs if c['status'] == 'generated'),
                              'output_tokens_generated': sum(c['output_tokens'] for c in costs),
                              'input_rejections': sum(c['status'] == 'input_rejected' for c in costs),
                              'seconds_generation': sum(c['seconds'] for c in costs)},
                    'evaluation_material_accessed': False, 'scalar_scores_created': False}
        names = ['plan.json', 'protocol.json', 'model_identity.json', 'source_identity.json', 'complete.json']
        names += [str((cases / data.text_hash(s.id) / name).relative_to(output))
                  for s in samples for name in ['case.json', 'seal.json']]
        names += cache_names
        if (output / 'complete.json').exists():
            if data.read_json(output / 'complete.json') != complete:
                raise ValueError('Existing completion changed; explicit recovery required')
        else:
            data.write_json(output / 'complete.json', complete)
        if (output / 'seal.json').exists():
            data.verify_seal(output, expected=set(names))
        else:
            _seal(output, names)
        return complete
    finally:
        lock.unlink()


def _available_ram():
    if os.name == 'nt':
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ['total_physical', 'available_physical', 'total_page', 'available_page',
                 'total_virtual', 'available_virtual', 'available_extended']]
        status = MemoryStatus(); status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise RuntimeError('Cannot establish available physical RAM')
        return status.available_physical
    return os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')


class HFGenerator:
    """Explicitly admitted local Qwen-3B NF4 instrument; no imports at construction.

    The 3-GiB limit is the PyTorch CUDA allocator limit. External CUDA library
    allocations are not a system-wide reservation; allocation failure stops the run.
    """

    def __init__(self, model, model_identity, *, admit_resources=False):
        self.model_path = Path(model).resolve()
        self.model_identity = _model_identity(model_identity)
        if self.model_identity['model_id'] != 'Qwen/Qwen2.5-3B-Instruct':
            raise ValueError('This resource-qualified harness supports only the frozen Qwen-3B instrument')
        if self.model_identity.get('runtime') != HF_RUNTIME:
            raise ValueError('HF identity must bind the exact CUDA/transformers/accelerate/bitsandbytes runtime')
        self.admitted, self.tokenizer, self.model = admit_resources, None, None

    def verify_frozen_identity(self):
        files = self.model_identity['model_files_sha256']
        paths = []
        for name, digest in files.items():
            path = (self.model_path / name).resolve()
            if (Path(name).is_absolute() or str(Path(name)) != name or
                    any(p in {'.', '..'} for p in Path(name).parts) or not path.is_relative_to(self.model_path)
                    or path.name.casefold() in ANNOTATION_NAMES):
                raise ValueError('Unsafe model manifest path')
            paths.append((path, digest))
        if not {'config.json', 'tokenizer_config.json', 'tokenizer.json',
                'model.safetensors.index.json'} <= set(files):
            raise ValueError('Frozen model manifest is incomplete')
        if {str(p.relative_to(self.model_path)) for p in self.model_path.rglob('*') if p.is_file()} != set(files):
            raise ValueError('Frozen model directory has missing or unbound files')
        for path, digest in paths:
            if data.file_hash(path) != digest:
                raise ValueError('Frozen model file changed: ' + path.name)
        index = data.read_json(self.model_path / 'model.safetensors.index.json')
        if not set(index['weight_map'].values()) <= set(files):
            raise ValueError('Model weight shards are absent from frozen manifest')

    def _prepare(self):
        if self.tokenizer is not None:
            return
        if not self.admitted:
            raise RuntimeError('HF resource admission must be explicit before tokenizer/model loading')
        if _available_ram() < 2 * 1024**3:
            raise RuntimeError('Resource admission rejected: available RAM below 2 GiB')
        if any(version(package) != pin for package, pin in HF_RUNTIME.items()):
            raise RuntimeError('Use the separately prepared pinned CUDA/transformers/accelerate/bitsandbytes environment')
        self.verify_frozen_identity()
        import torch
        from transformers import AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError('Resource admission requires CUDA')
        free, total = torch.cuda.mem_get_info(0)
        if free < 3 * 1024**3:
            raise RuntimeError('Resource admission rejected: free CUDA memory below 3 GiB')
        torch.cuda.set_per_process_memory_fraction(3 * 1024**3 / total, 0)
        self.admission = {'available_ram_bytes': _available_ram(), 'free_cuda_bytes': free,
                          'cuda_allocator_limit_bytes': 3 * 1024**3, 'device': 'cuda:0'}
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True,
                                                       trust_remote_code=False)

    def count_input_tokens(self, messages):
        self._prepare()
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True))

    def __call__(self, messages, *, max_new_tokens):
        if type(max_new_tokens) is not int or max_new_tokens not in {
                DEFAULT_PROTOCOL['proposer_max_new_tokens'], DEFAULT_PROTOCOL['observation_max_new_tokens']}:
            raise ValueError('HF generation budget must be one of the frozen protocol limits')
        self._prepare()
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        if self.model is None:
            if _available_ram() < 2 * 1024**3 or torch.cuda.mem_get_info(0)[0] < 3 * 1024**3:
                raise RuntimeError('Resources changed before model load; explicit recovery required')
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, local_files_only=True, trust_remote_code=False,
                device_map={'': 0}, attn_implementation='sdpa', torch_dtype=torch.float16,
                quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16),
                low_cpu_mem_usage=True)
            self.model.eval()
        encoded = self.tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, return_tensors='pt').to('cuda:0')
        if encoded.shape[-1] > DEFAULT_PROTOCOL['max_input_tokens']:
            raise RuntimeError('Input token guard changed; truncation is forbidden')
        with torch.inference_mode():
            generated = self.model.generate(encoded, attention_mask=torch.ones_like(encoded),
                max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
                temperature=None, top_p=None, top_k=None, use_cache=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)
        tokens = generated[0, encoded.shape[-1]:].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        return {'text': self.tokenizer.decode(tokens, skip_special_tokens=True),
                'raw_text': self.tokenizer.decode(tokens, skip_special_tokens=False),
                'output_token_ids': tokens, 'input_tokens': encoded.shape[-1],
                'output_tokens': len(tokens), 'finish_reason': 'eos' if tokens and tokens[-1] in eos
                    else 'max_new_tokens', 'resource_admission': self.admission}
