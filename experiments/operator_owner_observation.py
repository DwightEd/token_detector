"""Frozen H018 bool-owner controls, reusing the original local-effect kernel.

Preparation encodes only two constructed source prompts. Execution consumes
their sealed axes, never tokenizes, and never reruns the original S0 observer.
Neither stage reads annotations or promotes a natural-data detector.
"""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import time

import numpy as np

from . import operator_binding_observation as original
from .data import file_hash, read_json, text_hash, verify_seal, write_json
from .provenance import completed_identity, research_identity
from .relation_cache_qualification import capture_identity
from .relation_intervention import _seal, head_prerequisite
from .research.operator_owner_swap import prepare_swaps
from .research.streamed_observer import ResourceLimits, SafeTensorStore, StreamedObserver
from .streamed_qualification import _resources, verify_assets


WORLD_ORDER = ('target', 'control')
ORIGINAL_ID = '5684'
BUDGET = dict(model_calls=2, head_calls=8, extra_mlp_calls=256,
              extra_norm_calls=256, final_norm_replays=8,
              partition_o_calls=192, av_replays=64)
ARGUMENTS = ('protocol', 'axes', 'prepared', 'frontend_contract', 'baseline',
             'model', 'asset_manifest', 'asset_verification', 'asset_state',
             'head_qualification', 'capture_root', 'device')
ADDED_MODULES = {'experiments/operator_owner_observation.py',
                 'experiments.research/operator_owner_swap.py'}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _new_output(args):
    output = Path(args.output).resolve()
    for name in ('baseline', 'axes', 'prepared', 'model', 'capture_root', 'head_qualification', 'plan'):
        value = getattr(args, name, None)
        if value is not None:
            _require(not output.is_relative_to(Path(value).resolve()),
                     'New output must not mutate an input/prerequisite tree: ' + name)
    output.mkdir(parents=True, exist_ok=False)
    return output


def _protocol(args):
    protocol = read_json(args.protocol)
    _require(protocol.get('schema') == 'h018-operator-owner-phase2-v1'
             and protocol.get('world_order') == list(WORLD_ORDER)
             and protocol.get('budget') == BUDGET, 'Finite owner-control protocol changed')
    for path, digest in protocol['scientific_bindings'].items():
        _require(file_hash(Path(path)) == digest, 'Scientific admission/resource binding changed')
    return protocol


def _source_compatibility(old, current):
    """All old package bytes stay frozen except the explicit CLI dispatch."""
    expected, actual = old['source_sha256'], current['source_sha256']
    _require(set(actual) - set(expected) <= ADDED_MODULES,
             'An unadmitted package module was added after the original baseline')
    unchanged = {}
    for name, digest in expected.items():
        if name == 'experiments/run.py':
            continue
        _require(actual.get(name) == digest, 'Original numerical dependency changed: ' + name)
        unchanged[name] = digest
    for key in set(old) | set(current):
        if key != 'source_sha256':
            _require(old.get(key) == current.get(key), 'Original numerical runtime changed: ' + key)
    return unchanged


def _member(seal, name):
    matches = [digest for path, digest in seal.items() if path.replace('\\', '/') == name]
    _require(len(matches) == 1, 'Required baseline member missing/ambiguous: ' + name)
    return matches[0]


def baseline_reference(args, protocol, identity, *, full):
    """Hash the old complete root once in prepare; execution binds its metadata.

    The new forward never consumes S0 numeric arrays. No old raw file is copied.
    A later comparison must retain this original seal/audit provenance.
    """
    root = Path(args.baseline)
    seal = verify_seal(root) if full else read_json(root/'seal.json')
    required = ('plan.json', 'result.json', 'case-5684/seal.json',
                'case-5684/axis.json', 'case-5684/result.json', 'case-5684/prediction.json')
    hashes = {name: _member(seal, name) for name in required}
    for name in required:
        _require(file_hash(root/name) == hashes[name], 'Original baseline metadata changed: ' + name)
    plan, result = read_json(root/'plan.json'), read_json(root/'result.json')
    case, axis = read_json(root/'case-5684/result.json'), read_json(root/'case-5684/axis.json')
    frontend = read_json(args.frontend_contract)
    entry = next(row for row in frontend['cases'] if row['id'] == ORIGINAL_ID)
    _require(result.get('completed') is True and result.get('annotation_reads') == 0
             and all(result.get('actual_counts', {}).get(key) == value for key, value in BUDGET.items())
             and result['source_runtime'] == plan['source_runtime']
             and case.get('completed') is True and case.get('original_targets') == 233
             and case.get('annotation_reads') == 0
             and (axis['id'], axis['P'], axis['T'], axis['N']) == ('5684', 803, 233, 1036)
             and axis['original_response_sha256'] == entry['response_sha256'],
             'Original completed two-response baseline contract differs')
    _require(plan['axes_seal_sha256'] == file_hash(Path(args.axes)/'seal.json')
             and plan['frontend_sha256'] == file_hash(args.frontend_contract)
             and plan['inputs_sha256'] == file_hash(Path(args.prepared)/'inputs.jsonl'),
             'Original input/axes/frontend prerequisite differs')
    dependencies = _source_compatibility(plan['source_runtime'], identity)
    audit_pin = protocol['baseline_audit']
    audit_root = Path(audit_pin['root'])
    _require(file_hash(audit_root/'seal.json') == audit_pin['seal_sha256']
             and file_hash(audit_root/'audit.json') == audit_pin['audit_sha256'],
             'Original independent audit pin differs')
    verify_seal(audit_root, required=('audit.json', 'spec.json'))
    audit, spec = read_json(audit_root/'audit.json'), read_json(audit_root/'spec.json')
    _require(audit.get('passed') is True and audit.get('complete_raw_audited') is True
             and audit.get('label_reads') == 0 and audit['cases']['5684']['passed'] is True
             and spec['observations_seal_sha256'] == file_hash(root/'seal.json')
             and spec['axes_seal_sha256'] == plan['axes_seal_sha256']
             and spec['frontend_sha256'] == plan['frontend_sha256'],
             'Independent audit does not bind the reused original baseline')
    return dict(schema='h018-original-baseline-reference-v1', original_id=ORIGINAL_ID,
        root=str(root), root_seal_sha256=file_hash(root/'seal.json'), members_sha256=hashes,
        audit=dict(audit_pin), numerical_source_sha256=dependencies,
        prerequisites={key: plan[key] for key in
            ('assets', 'observer_capture_identity', 'head_prerequisite')},
        S0_new_model_calls=0, copied_raw_bytes=0,
        verification_scope='Full old-root SHA check at preparation; sealed metadata only at execution')


def _prerequisites(args):
    assets = verify_assets(args.asset_manifest, args.model,
        verification=args.asset_verification, state=args.asset_state)
    observer = capture_identity(Path(args.capture_root), Path(args.model)/'config.json')
    head = head_prerequisite(Path(args.head_qualification), assets, observer)
    config = read_json(Path(args.model)/'config.json')
    _require((config['num_hidden_layers'], config['hidden_size'], config['vocab_size'])
             == (32, 4096, 128256), 'Original observer dimensions changed')
    return dict(assets=assets, observer_capture_identity=observer, head_prerequisite=head)


def _tokenizer(model):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model, use_fast=True,
        local_files_only=True, trust_remote_code=False)


class _TokenizerCounts:
    """Retain attempted/returned prompt work even if the frontend fails."""

    def __init__(self, tokenizer, counts):
        self.tokenizer, self.counts = tokenizer, counts

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def apply_chat_template(self, *args, **kwargs):
        self.counts['chat_render_attempts'] += 1
        value = self.tokenizer.apply_chat_template(*args, **kwargs)
        self.counts['chat_template_renders'] += 1
        return value

    def __call__(self, *args, **kwargs):
        self.counts['source_encode_attempts'] += 1
        value = self.tokenizer(*args, **kwargs)
        self.counts['source_encodings'] += 1
        return value

    def decode(self, *args, **kwargs):
        raise RuntimeError('No original response decoding is admitted')


def prepare(args):
    """Two source-only encodings after identity gates; zero model forwards."""
    output = _new_output(args)
    counts = dict(model_calls=0, head_calls=0, annotation_reads=0, source_encode_attempts=0,
        source_encodings=0, chat_render_attempts=0, chat_template_renders=0,
        response_tokenizations=0, response_decodes=0, S0_encodings=0)
    start = time.monotonic()
    try:
        identity, protocol = research_identity(), _protocol(args)
        cases = original.load_cases(args.axes, args.prepared, args.frontend_contract)
        case = cases[ORIGINAL_ID]
        reference = baseline_reference(args, protocol, identity, full=True)
        prerequisites = _prerequisites(args)
        _require(prerequisites == reference['prerequisites'],
                 'Original baseline weights/observer/head identity differs')
        contract = read_json(args.frontend_contract)
        tokenizer = _TokenizerCounts(_tokenizer(args.model), counts)
        counts['tokenizer_constructions'] = 1
        prepared = prepare_swaps(case['sample'], contract['source_fields'], tokenizer, case['axes'])
        del tokenizer
        audit = prepared['audit']
        _require(set(prepared['worlds']) == set(WORLD_ORDER) and audit.get('passed') is True
                 and audit.get('source_encodings') == 2 and audit.get('S0_encodings') == 0
                 and audit.get('response_tokenizations') == 0 and audit.get('response_decodes') == 0
                 and audit.get('model_calls') == 0 and audit.get('annotation_reads') == 0
                 and counts['source_encodings'] == counts['chat_template_renders'] == 2,
                 'Only two new source prompt encodings are admitted')
        frontend = output/'frontend'
        frontend.mkdir()
        entry = next(row for row in contract['cases'] if row['id'] == ORIGINAL_ID)
        entries = []
        for name in WORLD_ORDER:
            world = prepared['worlds'][name]
            _require(world['response'] == case['sample'].response
                     and (world['P'], world['T'], world['N']) == (803, 233, 1036),
                     'Constructed control changed the original response/dimensions')
            axes = world['axes']
            _require(all(isinstance(value, np.ndarray) for value in axes.values()),
                     'Only original numeric axes may enter the sealed axis archive')
            with (frontend/f'axes-{name}.npz').open('xb') as stream:
                np.savez(stream, **axes)
            record = {key: value for key, value in world.items() if key != 'axes'}
            record.update(original_id=ORIGINAL_ID, id=name, source_id=case['sample'].source_id,
                          constructed_counterfactual=True, native_collection_answer_index=8,
                          primitive_response_path=entry['primitive_response_path'],
                          primitive_response_sha256=entry['primitive_response_sha256'])
            write_json(frontend/f'world-{name}.json', record)
            entries.append(dict(entry, id=name, original_id=ORIGINAL_ID,
                constructed_counterfactual=True, source_sha256=text_hash(world['source']),
                source_fields=world['field_proofs']))
        cf = dict(schema='h018-owner-counterfactual-frontend-v1', cases=entries,
            original_id=ORIGINAL_ID, constructed_counterfactual=True, world_order=list(WORLD_ORDER),
            evidence_bindings=dict(contract['evidence_bindings']), audit=audit,
            original_frontend_sha256=file_hash(args.frontend_contract),
            semantic_scope='Fixed typed Boolean edits only; not natural benchmark labels or whole-response truth')
        write_json(frontend/'FRONTEND_CF.json', cf)
        write_json(frontend/'baseline-reference.json', reference)
        _seal(frontend)
        plan = dict(schema='h018-operator-owner-execution-plan-v1', protocol=protocol,
            source_runtime=completed_identity(identity), **prerequisites,
            frontend_sha256=file_hash(frontend/'FRONTEND_CF.json'),
            axes_seal_sha256=file_hash(frontend/'seal.json'), baseline_reference=reference,
            original_axes_seal_sha256=file_hash(Path(args.axes)/'seal.json'),
            original_frontend_sha256=file_hash(args.frontend_contract),
            inputs_sha256=file_hash(Path(args.prepared)/'inputs.jsonl'),
            cases={name: dict(P=803, T=233, N=1036) for name in WORLD_ORDER},
            args={key: str(getattr(args, key)) for key in ARGUMENTS},
            original_id=ORIGINAL_ID, constructed_counterfactual=True,
            model_calls=0, head_calls=0, annotation_reads=0, stage='prepare_only',
            tokenizer_counts=counts, finite_control_only=True, method_promotion=False, full900_admitted=False)
        write_json(output/'plan.json', plan)
        _seal(output)
        return plan
    except BaseException as error:
        write_json(output/'failure.json', dict(type=type(error).__name__, message=str(error),
            actual_counts=counts, automatic_retry=False, elapsed_seconds=time.monotonic()-start))
        _seal(output)
        raise


def _world_cases(plan_root, original_case):
    frontend = Path(plan_root)/'frontend'
    records = {}
    unchanged = set(original_case['axes']) - {'token_ids', 'full_offsets'}
    for name in WORLD_ORDER:
        record = read_json(frontend/f'world-{name}.json')
        with np.load(frontend/f'axes-{name}.npz', allow_pickle=False) as saved:
            axes = {key: saved[key] for key in saved.files}
        _require(record['id'] == name and record['original_id'] == ORIGINAL_ID
                 and record['constructed_counterfactual'] is True
                 and record['native_collection_answer_index'] == 8
                 and (record['P'], record['T'], record['N']) == (803, 233, 1036)
                 and record['response'] == original_case['sample'].response
                 and record['response_sha256'] == text_hash(original_case['sample'].response)
                 and record['source_sha256'] == text_hash(record['source'])
                 and record['prompt_sha256'] == text_hash(record['prompt'])
                 and set(axes) == set(original_case['axes'])
                 and all(axes[key].dtype == original_case['axes'][key].dtype
                         and np.array_equal(axes[key], original_case['axes'][key]) for key in unchanged)
                 and axes['token_ids'].dtype == axes['full_offsets'].dtype == np.int64
                 and np.array_equal(axes['token_ids'][803:], original_case['axes']['answer_ids']),
                 'Sealed constructed world changed its original-response identity')
        sample = replace(original_case['sample'], id=name, source=record['source'], prompt=record['prompt'])
        sample.inputs  # Unique source occurrence; no new question/response.
        records[name] = dict(sample=sample, response=original_case['response'],
            axis=original_case['axis'], metadata=record, axes=axes, P=803, T=233, N=1036)
    return records


@contextmanager
def _open_observer(args, assets, output):
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats(args.device)
    resources = _resources(torch.device(args.device))
    resources.update(torch=torch.__version__, cuda_runtime=torch.version.cuda,
        device_capability=list(torch.cuda.get_device_capability(args.device)),
        device_name=torch.cuda.get_device_name(args.device))
    write_json(output/'post-import-resource-admission.json', resources)
    _require(torch.__version__ == '2.8.0+cu128' and torch.version.cuda == '12.8'
             and resources['device_capability'] == [12, 0], 'Frozen SM120/Torch runtime changed')
    if (resources['available_physical_ram_bytes'] < 512*1024**2
            or resources.get('gpu_free_bytes', 0) < 512*1024**2):
        raise MemoryError('Fixed post-import host/GPU reserve not met')
    files = assets['files']
    store = SafeTensorStore(args.model,
        index_sha256=files['model.safetensors.index.json']['sha256'], expected_files=files)
    with StreamedObserver(store, read_json(Path(args.model)/'config.json'),
            config_sha256=files['config.json']['sha256'],
            limits=ResourceLimits(max_sequence_tokens=1036), device=args.device) as observer:
        yield observer


def observe_worlds(observer, cases, output, counts):
    """Exactly two new constructed worlds; old S0 is never a model argument."""
    _require(tuple(cases) == WORLD_ORDER, 'Only target/control in frozen order are admitted')
    results = {}
    for name in WORLD_ORDER:
        results[name] = original.observe_case(observer, cases[name], output/f'case-{name}', counts)
        write_json(output/f'progress-{name}.json', dict(completed=list(results), actual_counts=dict(counts)))
    return results


def run(args):
    """Explicit sealed preparation or separately claimed finite execution."""
    if args.prepare_only:
        _require(args.plan is None, 'Preparation cannot consume an execution plan')
        return prepare(args)
    _require(args.plan is not None, 'Execution requires a separately sealed plan')
    plan_root = Path(args.plan)
    verify_seal(plan_root, required=('plan.json',))
    plan, identity = read_json(plan_root/'plan.json'), research_identity()
    _require(plan['schema'] == 'h018-operator-owner-execution-plan-v1'
             and identity == plan['source_runtime'], 'Prepared source/runtime changed')
    _require(all(str(getattr(args, key)) == value for key, value in plan['args'].items()),
             'Frozen execution argument changed')
    protocol = _protocol(args)
    _require(protocol == plan['protocol'], 'Frozen finite protocol changed')
    _require(file_hash(plan_root/'frontend/seal.json') == plan['axes_seal_sha256']
             and file_hash(plan_root/'frontend/FRONTEND_CF.json') == plan['frontend_sha256'],
             'Constructed frontend/axes binding changed')
    verify_seal(plan_root/'frontend', required=('FRONTEND_CF.json', 'baseline-reference.json',
        'world-target.json', 'world-control.json', 'axes-target.npz', 'axes-control.npz'))
    old_cases = original.load_cases(args.axes, args.prepared, args.frontend_contract)
    _require(baseline_reference(args, protocol, identity, full=False) == plan['baseline_reference'],
             'Original baseline reference changed after preparation')
    prerequisites = _prerequisites(args)
    _require(all(plan[key] == value for key, value in prerequisites.items()),
             'Original assets/observer/head prerequisite changed')
    _require(prerequisites == plan['baseline_reference']['prerequisites'],
             'Current weights/observer/head no longer match the original baseline')
    cases = _world_cases(plan_root, old_cases[ORIGINAL_ID])
    output = _new_output(args)
    counts = {key: 0 for key in BUDGET}
    counts.update(host_transfer_bytes=0, max_host_staging_bytes=0)
    write_json(output/'plan.json', plan)
    write_json(output/'constructed-worlds.json', dict(original_id=ORIGINAL_ID,
        constructed_counterfactual=True, worlds=list(WORLD_ORDER), S0_new_model_calls=0,
        finite_control_only=True, not_natural_benchmark_predictions=True))
    start = time.monotonic()
    results = {}
    try:
        with _open_observer(args, plan['assets'], output) as observer:
            results = observe_worlds(observer, cases, output, counts)
            _require(all(counts[key] == value for key, value in BUDGET.items()),
                     'Actual new call counts differ from the finite budget')
            resources = observer.resource_report()
        report = dict(completed=True, phase='finite_owner_control_only', actual_counts=counts,
            cases=results, resource_report=resources, elapsed_seconds=time.monotonic()-start,
            source_runtime=completed_identity(identity), annotation_reads=0, auroc=None,
            original_id=ORIGINAL_ID, constructed_counterfactual=True,
            phase2_admitted=True, finite_control_only=True, method_promotion=False,
            full900_admitted=False, detector_validated=False, automatic_retry=False,
            baseline_reference=plan['baseline_reference'], S0_new_model_calls=0,
            child_phase2_false_scope='Frozen child algorithm flag means no general promotion; this finite control is admitted')
        write_json(output/'result.json', report)
        _seal(output)
        return report
    except BaseException as error:
        # A sealed unit can complete before its progress file fails to publish.
        # Bind only that small result/seal here; raw auditing remains separate.
        completed, completion_evidence, completion_errors = [], {}, {}
        for name in WORLD_ORDER:
            child = output/f'case-{name}'
            if not (child/'result.json').exists() or (child/'failure.json').exists():
                continue
            try:
                seal = read_json(child/'seal.json')
                digest = file_hash(child/'result.json')
                _require(_member(seal, 'result.json') == digest
                         and read_json(child/'result.json').get('completed') is True,
                         'Child completion is not bound by its seal')
                completed.append(name)
                completion_evidence[name] = dict(result_sha256=digest,
                    seal_sha256=file_hash(child/'seal.json'))
            except Exception as validation_error:
                completion_errors[name] = dict(type=type(validation_error).__name__,
                                               message=str(validation_error))
        write_json(output/'failure.json', dict(type=type(error).__name__, message=str(error),
            actual_counts=dict(counts), completed=completed, annotation_reads=0,
            completion_evidence=completion_evidence, completion_validation_errors=completion_errors,
            completion_scope='Completed child result bound by its child seal; raw audit remains separate',
            automatic_retry=False, S0_new_model_calls=0,
            elapsed_seconds=time.monotonic()-start))
        _seal(output)
        raise
