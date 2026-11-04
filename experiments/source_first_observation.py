"""Separately sealed source-only inventory, resource preflight and proposals.

This instrument never accepts an existing response or produces detector scores.
The shared HFGenerator owns model loading and decoding; hooks only observe calls.
"""
import copy
import json
import os
from pathlib import Path
import re
import shutil
import time
import traceback

from . import data
from .provenance import completed_identity, research_identity
from .semantic_qualification import HFGenerator, _available_ram, _seal


ARGUMENTS = ('source', 'protocol', 'system_prompt', 'schema', 'admission',
             'review_root', 'model_identity', 'model')
MAX_SECONDS = 1800
MAX_OUTPUT_BYTES = 64 * 1024**2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _arguments(args):
    return {key: str(Path(getattr(args, key)).resolve()) for key in ARGUMENTS}


def _seal_tree(root):
    _seal(root, [str(p.relative_to(root)) for p in sorted(root.rglob('*'))
                 if p.is_file() and p != root / 'seal.json'])


def _budget(root, started):
    _require(time.monotonic() - started <= MAX_SECONDS, 'Finite execution time limit exceeded')
    _require(shutil.disk_usage(root).free >= 5 * 1024**3, 'Research data disk reserve below5GiB')
    _require(sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) <= MAX_OUTPUT_BYTES,
             'Finite metadata output limit exceeded')


def _new_output(args, protocol):
    root = Path(args.output).resolve()
    frozen_root = Path(protocol['output_root']).resolve()
    _require(root != frozen_root and root.is_relative_to(frozen_root),
             'H020 output must be a new child of the frozen unified data root')
    for key in ('model', 'plan', 'preflight'):
        value = getattr(args, key, None)
        if value is not None:
            _require(not root.is_relative_to(Path(value).resolve()),
                     'Output must not mutate an input tree: ' + key)
    root.mkdir(parents=True, exist_ok=False)
    return root


def _contract(args):
    protocol, admission = data.read_json(args.protocol), data.read_json(args.admission)
    _require(protocol.get('schema') == 'h020-source-first-record-proposal-v1'
             and admission.get('status') == 'finite_coding_and_zero_model_M0_admitted'
             and admission.get('hypothesis_id') == protocol.get('hypothesis_id'),
             'Independent finite scientific admission is required')
    bindings = admission['scientific_bindings']
    _require(len(bindings) >= 6, 'Missing independent scientific review')
    for binding in bindings:
        path = Path(args.review_root) / binding['path']
        _require(data.file_hash(path) == binding['sha256'], 'Scientific contract changed: ' + str(path))
    admitted = {Path(row['path']).name: row['sha256'] for row in bindings}
    for key in ('protocol', 'system_prompt', 'schema'):
        path = Path(getattr(args, key))
        _require(admitted.get(path.name) == data.file_hash(path), 'Actual wire differs from admitted contract')
    _require(data.file_hash(Path(args.source)) == protocol['source_input_sha256'], 'Source-only roster changed')
    _require(data.file_hash(Path(args.model_identity)) == protocol['weights_runtime_identity_sha256'],
             'Frozen weight/runtime identity changed')
    _require(protocol['max_generation_requests'] == 6 and protocol['generation']['max_input_tokens'] == 4096
             and protocol['generation']['max_new_tokens'] == 1536 and protocol['automatic_retry'] is False,
             'Finite instrument limits differ')
    return protocol


def _wire(args, cases):
    prompt = Path(args.system_prompt).read_text(encoding='utf-8')
    blocks = re.findall(r'```text\n(.*?)\n```', prompt, flags=re.S)
    _require(len(blocks) == 1, 'Fixed system text block missing or ambiguous')
    schema = Path(args.schema).read_text(encoding='utf-8')
    match = re.search(r'```json\n(.*?)\n```\s*\n([^\n]+)', schema, flags=re.S)
    _require(match is not None, 'Fixed schema and immediate array legend missing')
    enum = ('Vertical bars denote alternatives; output exactly one option, never the pipe-separated literal. '
            'All keys are required; arrays may be empty when unused. Example entries illustrate shapes, '
            'not source facts; every reference must point to an existing entry.')
    _require(enum in schema, 'Fixed enum wire clarification missing')
    schema_wire = enum + '\n' + match.group(1) + '\n' + match.group(2)
    messages = []
    for case in cases:
        payload = dict(case['model_data'])
        _require(list(payload) == ['task_prefix', 'task_suffix', 'passages', 'anchors'],
                 'Model payload contains undeclared fields or metadata')
        payload['schema'] = schema_wire
        messages.append([{'role': 'system', 'content': blocks[0]},
                         {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False,
                            separators=(',', ':'), sort_keys=False)}])
    return messages


def prepare(args):
    from .research.source_first_inventory import prepare_inventory
    protocol = _contract(args)
    source_runtime = research_identity()
    records = [json.loads(line) for line in Path(args.source).read_text(encoding='utf-8').splitlines()
               if line.strip()]
    inventory = prepare_inventory(records, protocol)
    _require(len(inventory['cases']) == 6, 'All six fixed source cases are required')
    messages = _wire(args, inventory['cases'])
    root = _new_output(args, protocol)
    try:
        plan = dict(schema='h020-source-execution-plan-v1', arguments=_arguments(args), protocol=protocol,
                    source_runtime=source_runtime, inventory=inventory, messages=messages,
                    model_inputs_contain_R=False, semantic_licensed=False,
                    planned_requests=6, actual_model_requests=0, tokenizer_calls=0)
        data.write_json(root / 'plan.json', plan)
        data.write_json(root / 'result.json', dict(completed=True, stage='M0_source_only_plan',
                        cases=6, actual_model_requests=0, tokenizer_calls=0, auroc=None,
                        source_runtime=completed_identity(source_runtime)))
        _seal_tree(root)
        return plan
    except BaseException as error:
        _failure(root, error, {})
        raise


def _load_plan(args):
    _require(args.plan is not None, 'A separately sealed M0 plan is required')
    root = Path(args.plan).resolve()
    data.verify_seal(root, required=('plan.json', 'result.json'))
    plan = data.read_json(root / 'plan.json')
    _require(data.read_json(root / 'result.json')['completed'] is True
             and plan['schema'] == 'h020-source-execution-plan-v1'
             and plan['arguments'] == _arguments(args) and plan['protocol'] == _contract(args)
             and plan['source_runtime']['source_sha256'] == research_identity()['source_sha256'],
             'Sealed source plan, scientific contract or installed source differs')
    return plan


def _failure(root, error, accounting):
    data.write_json(root / 'failure.json', dict(completed=False, type=type(error).__name__,
                    message=str(error), traceback=traceback.format_exc(), accounting=accounting,
                    automatic_retry=False, partial_generated_tokens='unknown_without_backend_return',
                    detector_validated=False, auroc=None))
    _seal_tree(root)


class ForwardAccounting:
    """Read-only root-module hooks; never replace a model's forward signature."""
    def __init__(self, model, journal, expected_input_ids, started):
        self.model, self.journal = model, journal
        self.expected = expected_input_ids
        self.started = started
        self.attempts = self.returns = self.input_positions = 0
        self.rows, self.handles = [], []

    def _append(self, row):
        self.journal.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        self.journal.flush()

    def before(self, module, args, kwargs):
        self.attempts += 1
        ids = kwargs.get('input_ids', args[0] if args else None)
        shape = tuple(ids.shape) if ids is not None else ()
        row = dict(event='forward_entry', ordinal=self.attempts, shape=list(shape),
                   logits_to_keep=kwargs.get('logits_to_keep'), returned=False)
        self.rows.append(row)
        self._append(row)
        _require(time.monotonic() - self.started <= MAX_SECONDS, 'Decoder execution time limit exceeded')
        _require(len(shape) == 2 and shape[0] == 1 and shape[1] > 0,
                 'Unexpected root model input shape')
        _require(kwargs.get('logits_to_keep') == 1, 'Pinned last-logit resource path was not selected')
        values = ids.tolist()[0]
        row['input_token_ids'] = values
        row['phase'] = 'prefill' if self.attempts == 1 else 'decode'
        if self.attempts == 1:
            _require(values == self.expected, 'Actual prefill token IDs differ from sealed preflight')
        else:
            _require(shape[1] == 1, 'Pinned greedy path unexpectedly batches/chunks decoder input')
        self._append(dict(event='forward_input', ordinal=self.attempts, token_ids=values,
                          phase=row['phase']))

    def after(self, module, args, kwargs, output):
        returned = output is not None
        if returned:
            self.returns += 1
            self.input_positions += self.rows[-1]['shape'][1]
            self.rows[-1]['returned'] = True
        self._append(dict(event='forward_exit', ordinal=self.attempts, python_returned=returned))

    def __enter__(self):
        self.handles.append(self.model.register_forward_pre_hook(self.before, with_kwargs=True))
        self.handles.append(self.model.register_forward_hook(self.after, with_kwargs=True, always_call=True))
        return self

    def __exit__(self, *_):
        for handle in reversed(self.handles):
            handle.remove()

    def report(self):
        return dict(forward_entry_attempts=self.attempts, successful_python_returns=self.returns,
                    processed_input_positions=self.input_positions,
                    python_returns_are_not_cuda_kernel_count=True, rows=self.rows)


class HFSourceInstrument:
    """Add resource and call accounting around the unchanged shared generator."""
    def __init__(self, model, identity):
        self.generator = HFGenerator(model, identity, admit_resources=True)
        self.tokenizer_calls = 0
        self.resources = []

    def snapshot(self, stage, *, cuda=False):
        row = dict(stage=stage, available_ram_bytes=_available_ram())
        if cuda:
            import torch
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info(0)
            row.update(free_cuda_bytes=free, total_cuda_bytes=total,
                       cuda_allocated_bytes=torch.cuda.memory_allocated(0),
                       cuda_reserved_bytes=torch.cuda.memory_reserved(0),
                       cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                       cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(0))
        self.resources.append(row)
        return row

    def input_ids(self, messages):
        self.tokenizer_calls += 1
        count = self.generator.count_input_tokens(messages)
        self.tokenizer_calls += 1
        ids = self.generator.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        _require(len(ids) == count, 'Independent tokenizer count and exact input IDs differ')
        pad = self.generator.tokenizer.pad_token_id or self.generator.tokenizer.eos_token_id
        _require(pad == 151643, 'Actual shared-generator padding token differs')
        return ids

    def load(self):
        self.generator._ensure_model()
        from transformers import GenerationConfig
        model = self.generator.model
        _require(getattr(model, 'is_loaded_in_4bit', False) is True
                 and model.config._attn_implementation == 'sdpa', 'Actual model load contract differs')
        config = copy.deepcopy(model.generation_config.to_dict())
        expected = GenerationConfig.from_dict(data.read_json(
            self.generator.model_path / 'generation_config.json')).to_dict()
        _check_generation_configuration(config, expected)
        _require(config['repetition_penalty'] == 1.05 and config['eos_token_id'] == [151645, 151643]
                 and config['num_return_sequences'] == 1,
                 'Inherited generation configuration differs')
        return dict(base_generation_config=config, explicit_shared_generator_kwargs=dict(
            max_new_tokens=1536, do_sample=False, num_beams=1, temperature=None, top_p=None,
            top_k=None, use_cache=True, pad_token_id=151643),
            interpretation='deterministic_greedy_after_fixed_logits_processors')

    def generate(self, messages, ids, root, started):
        journal_path = root / 'forward-events.jsonl'
        with journal_path.open('x', encoding='utf-8') as journal:
            counter = ForwardAccounting(self.generator.model, journal, ids, started)
            try:
                with counter:
                    self.tokenizer_calls += 1
                    raw = self.generator(messages, max_new_tokens=1536)
                data.write_json(root / 'backend_raw.json', raw)
                return raw
            finally:
                data.write_json(root / 'forward-accounting.json', counter.report())


def _identity(args):
    old = data.read_json(args.model_identity)
    return {key: old[key] for key in ('model_id', 'official_revision', 'model_files_sha256', 'runtime')}


def _check_generation_configuration(observed, expected):
    # Compare the complete pinned-runtime configuration derived from the frozen
    # file, including unknown future processor/cache/stop fields. Only Hub/version
    # metadata are ignored; _from_model_config is retained because it affects
    # legacy configuration fallback. No output-dependent tolerance is used.
    metadata = {'_commit_hash', 'transformers_version'}
    left = {key: value for key, value in observed.items() if key not in metadata}
    right = {key: value for key, value in expected.items() if key not in metadata}
    differences = {key: dict(observed=left.get(key), expected=right.get(key),
                             observed_present=key in left, expected_present=key in right)
                   for key in left.keys() | right.keys() if key not in left or key not in right
                   or left[key] != right[key]}
    _require(not differences, 'Unbound generation processors/stop/cache configuration: '
             + json.dumps(differences, sort_keys=True, ensure_ascii=False, default=repr))


def _resource_gate(instrument, stage, protocol, *, preimport=False):
    row = instrument.snapshot(stage, cuda=not preimport)
    gates = protocol['resource_gates_bytes']
    host = gates['pre_import_available_ram' if preimport else
                 'post_import_tokenizer_and_preload_and_prerequest_available_ram']
    _require(row['available_ram_bytes'] >= host, 'Available host RAM below frozen resource floor at ' + stage)
    if not preimport:
        _require(row['free_cuda_bytes'] >= gates['post_import_tokenizer_and_preload_and_prerequest_free_cuda'],
                 'Free CUDA memory below frozen resource floor at ' + stage)
    return row


def _tokenize(instrument, plan):
    rows = []
    for index, messages in enumerate(plan['messages']):
        ids = instrument.input_ids(messages)
        rows.append(dict(index=index, world_key=plan['inventory']['cases'][index]['world_key'],
                         input_token_ids=ids, input_tokens=len(ids), eligible=len(ids) <= 4096,
                         status='eligible' if len(ids) <= 4096 else 'input_limit_unknown_no_truncation'))
    _require(sum(row['input_tokens'] for row in rows if row['eligible']) <= 24576,
             'Total finite input budget exceeded')
    return rows


def preflight(args, *, instrument=None):
    plan = _load_plan(args)
    _require(args.admit_resources, 'Tokenization/CUDA preparation requires explicit resource admission')
    protocol = plan['protocol']
    root = _new_output(args, protocol)
    identity = research_identity()
    instrument = instrument or HFSourceInstrument(args.model, _identity(args))
    try:
        _resource_gate(instrument, 'pre_import', protocol, preimport=True)
        tokens = _tokenize(instrument, plan)
        _resource_gate(instrument, 'post_import_and_tokenizer', protocol)
        report = dict(completed=True, stage='M1_tokenizer_resource_preflight', tokens=tokens,
                      source_runtime=completed_identity(identity), model_requests=0,
                      tokenizer_calls=instrument.tokenizer_calls, resources=instrument.resources,
                      source_plan_seal_sha256=data.file_hash(Path(args.plan) / 'seal.json'),
                      source_protocol_sha256=data.file_hash(Path(args.protocol)),
                      model_weight_identity=_identity(args), actual_model_loads=0,
                      token_counts_are_not_model_load_admission=True, auroc=None)
        data.write_json(root / 'result.json', report)
        _seal_tree(root)
        return report
    except BaseException as error:
        _failure(root, error, dict(resources=instrument.resources,
                                  tokenizer_calls=instrument.tokenizer_calls, model_requests=0))
        raise


def _collect_prerequisites(args, plan, identity):
    _require(args.preflight is not None and args.ledger is not None and args.model_admission is not None,
             'Generation requires separate preflight, frozen source obligations and model admission')
    preflight_root = Path(args.preflight).resolve()
    data.verify_seal(preflight_root, required=('result.json',))
    preflight_result = data.read_json(preflight_root / 'result.json')
    ledger, admission = data.read_json(args.ledger), data.read_json(args.model_admission)
    _require(preflight_result['completed'] is True
             and preflight_result['stage'] == 'M1_tokenizer_resource_preflight'
             and preflight_result['source_plan_seal_sha256'] == data.file_hash(Path(args.plan) / 'seal.json')
             and preflight_result['source_runtime'] == identity,
             'Preflight does not bind this exact source/runtime/plan')
    _require(ledger.get('schema') == 'h020-source-only-obligation-ledger-v1'
             and ledger.get('frozen_before_model_outputs') is True
             and ledger.get('source_plan_seal_sha256') == preflight_result['source_plan_seal_sha256']
             and ledger.get('model_inputs_contain_R') is False,
             'Independently frozen source-only obligations are required')
    _require([(row['world_key'], row['prompt_sha256']) for row in ledger.get('cases', [])]
             == [(row['world_key'], row['prompt_sha256']) for row in plan['inventory']['cases']],
             'Source obligation ledger does not cover the exact six frozen source worlds')
    _require(admission.get('schema') == 'h020-model-execution-admission-v1'
             and admission.get('model_requests_admitted') == 6
             and admission.get('source_plan_seal_sha256') == preflight_result['source_plan_seal_sha256']
             and admission.get('preflight_seal_sha256') == data.file_hash(preflight_root / 'seal.json')
             and admission.get('ledger_sha256') == data.file_hash(Path(args.ledger))
             and admission.get('source_sha256') == identity['source_sha256']
             and data.file_hash(Path(admission['independent_code_review'])) == admission['independent_code_review_sha256'],
             'Separate exact source/model execution admission is missing or differs')
    return preflight_result


def collect(args, *, instrument=None):
    from .research.source_first_inventory import validate_proposal
    plan = _load_plan(args)
    identity = research_identity()
    preflight_result = _collect_prerequisites(args, plan, identity)
    _require(args.admit_resources, 'Model loading requires explicit resource admission')
    protocol = plan['protocol']
    root = _new_output(args, protocol)
    started = time.monotonic()
    instrument = instrument or HFSourceInstrument(args.model, _identity(args))
    accounting = dict(model_load_attempts=0, model_load_returns=0, backend_attempts=0,
                      backend_returns=0, returned_output_tokens=0, model_forward_entry_attempts=0,
                      model_successful_python_returns=0, processed_input_positions=0)
    try:
        _budget(root, started)
        _resource_gate(instrument, 'pre_import', protocol, preimport=True)
        tokens = _tokenize(instrument, plan)
        data.write_json(root / 'actual_tokenization.json', dict(tokens=tokens))
        _require(tokens == preflight_result['tokens'], 'Exact model messages/token IDs differ from preflight')
        _resource_gate(instrument, 'post_import_and_tokenizer', protocol)
        eligible = [row for row in tokens if row['eligible']]
        if eligible:
            _resource_gate(instrument, 'preload', protocol)
            accounting['model_load_attempts'] += 1
            effective = instrument.load()
            accounting['model_load_returns'] += 1
            data.write_json(root / 'generation_configuration.json', effective)
            instrument.snapshot('postload', cuda=True)
        completed = []
        for index, (case, messages, token) in enumerate(zip(plan['inventory']['cases'], plan['messages'], tokens)):
            _budget(root, started)
            child = root / ('request-' + str(index).zfill(2))
            child.mkdir()
            data.write_json(child / 'request.json', dict(index=index, world_key=case['world_key'],
                messages=messages, input_token_ids=token['input_token_ids'], input_tokens=token['input_tokens'],
                max_new_tokens=1536, source_only=True, semantic_licensed=False))
            if not token['eligible']:
                result = dict(status='input_limit_unknown_no_truncation', actual_backend_attempts=0)
            else:
                _resource_gate(instrument, 'prerequest-' + str(index), protocol)
                accounting['backend_attempts'] += 1
                raw = instrument.generate(messages, token['input_token_ids'], child, started)
                # Real instrument writes raw before checking. Fakes use this same publication boundary.
                if not (child / 'backend_raw.json').exists():
                    data.write_json(child / 'backend_raw.json', raw)
                accounting['backend_returns'] += 1
                _require(isinstance(raw, dict) and isinstance(raw.get('text'), str)
                         and isinstance(raw.get('raw_text'), str)
                         and raw.get('input_tokens') == token['input_tokens']
                         and type(raw.get('output_tokens')) is int and 0 < raw['output_tokens'] <= 1536
                         and len(raw.get('output_token_ids', [])) == raw['output_tokens']
                         and all(type(item) is int and item >= 0 for item in raw.get('output_token_ids', []))
                         and raw.get('finish_reason') in ('eos', 'max_new_tokens'),
                         'Shared generator returned inconsistent raw metadata')
                _require((raw['finish_reason'] == 'eos' and raw['output_token_ids'][-1] in [151645, 151643])
                         or (raw['finish_reason'] == 'max_new_tokens' and raw['output_tokens'] == 1536),
                         'Actual stop reason differs from the frozen generation limits')
                forward = data.read_json(child / 'forward-accounting.json')
                accounting['model_forward_entry_attempts'] += forward['forward_entry_attempts']
                accounting['model_successful_python_returns'] += forward['successful_python_returns']
                accounting['processed_input_positions'] += forward['processed_input_positions']
                _require(forward['forward_entry_attempts'] == forward['successful_python_returns'] == raw['output_tokens']
                         and forward['processed_input_positions'] == token['input_tokens'] + raw['output_tokens'] - 1,
                         'Actual root forward count differs from pinned greedy path')
                _require([row['input_token_ids'] for row in forward['rows'][1:]]
                         == [[token_id] for token_id in raw['output_token_ids'][:-1]],
                         'Actual decoder inputs differ from returned generated token IDs')
                accounting['returned_output_tokens'] += raw['output_tokens']
                _require(accounting['returned_output_tokens'] <= 9216, 'Total finite output budget exceeded')
                validation = validate_proposal(raw['text'], case, raw['finish_reason'])
                data.write_json(child / 'proposal_validation.json', validation)
                result = dict(status='proposal_returned_not_licensed', output_tokens=raw['output_tokens'],
                              finish_reason=raw['finish_reason'], mechanical_gate_passed=validation['mechanical_gate_passed'])
                instrument.snapshot('postrequest-' + str(index), cuda=True)
            data.write_json(child / 'result.json', result)
            _seal_tree(child)
            completed.append(dict(index=index, world_key=case['world_key'], **result))
        _budget(root, started)
        report = dict(completed=True, stage='M2_source_proposals_only', requests=completed,
                      accounting=accounting, tokenizer_calls=instrument.tokenizer_calls,
                      tokenizer_calls_are_attempts=True,
                      resources=instrument.resources, source_runtime=completed_identity(identity),
                      elapsed_seconds=time.monotonic() - started, semantic_licensed=False,
                      independent_source_only_review='required_not_run_here',
                      detector_validated=False, risk_outputs=0, annotation_reads=0, auroc=None,
                      automatic_retry=False)
        data.write_json(root / 'result.json', report)
        _seal_tree(root)
        return report
    except BaseException as error:
        # A failed generate may already have performed many forwards. Do not
        # replace those observed partial counts with zeros or the token budget.
        forward_reports = [data.read_json(path) for path in sorted(root.glob('request-*/forward-accounting.json'))]
        accounting['model_forward_entry_attempts'] = sum(row['forward_entry_attempts'] for row in forward_reports)
        accounting['model_successful_python_returns'] = sum(row['successful_python_returns'] for row in forward_reports)
        accounting['processed_input_positions'] = sum(row['processed_input_positions'] for row in forward_reports)
        accounting['resources'] = instrument.resources
        accounting['tokenizer_calls'] = instrument.tokenizer_calls
        _failure(root, error, accounting)
        raise


def run(args):
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        _require(os.environ.get(name) == '2', 'Finite source instrument requires exactly2 threads')
    _require(os.environ.get('TOKENIZERS_PARALLELISM') == 'false', 'Tokenizer parallelism must be disabled')
    if args.source_stage == 'prepare':
        return prepare(args)
    if args.source_stage == 'preflight':
        return preflight(args)
    _require(args.source_stage == 'collect', 'Unknown source-only stage')
    return collect(args)
