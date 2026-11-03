"""Full original-response NLI comparator, with sealed tokenizer-only planning.

The external model and numerical kernels live in nli_qualification. This module
only streams the frozen original sentence inventory into the mainline token
prediction format. It never reads annotations or computes evaluation metrics.
"""

import json
import os
from pathlib import Path

import numpy as np

from myproject.measure import answer_axis, token_scores
from . import data, nli_qualification as nli, pipeline
from .datasets import read_samples, read_jsonl
from .provenance import research_identity, completed_identity


DEFAULT_PROTOCOL = {
    'schema': 'local-nli-dataset-protocol-v1',
    'purpose': 'frozen_external_supervised_full_population_comparator',
    'task': 'Data2txt', 'split': 'test', 'expected_samples': 900, 'expected_sources': 150,
    'expected_sentence_units': 8240, 'expected_measurement_units': 8247,
    'expected_original_valid_targets': 178343,
    'original_inputs_sha256': '050c79f2711aa292845f27ed4f3ae8b68f0d263bc9d29678486f7e8fb6477521',
    'original_manifest_sha256': 'd1f0ba0658222556e2542e145da84538d14a9fa5b230dd9bb8fa34b830944b80',
    'inventory_sha256': '709e4ad4e339b42e5badcbc9f7e7c4aa365cb8f1eff5f705e825c622977be0a1',
    'asset_inventory_sha256': '8bef1cfadbf591074c4096ff8b6c0430427ddeda40f988507cd5ba4a801d6da4',
    'model_id': 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',
    'model_revision': '6f5cf0a2b59cabb106aca4c287eed12e357e90eb',
    'execution_mode': 'dataset-fp32-batch1-v1', 'dtype': 'float32',
    'attention_implementation': 'eager', 'device': 'cuda:0', 'batch_size': 1,
    'max_pair_tokens': 512, 'pair_special_tokens': 3, 'overlap_rule': 'min(64,budget//4)',
    'source_policy': 'exact-original-source-all-token-windows-source-first-no-truncation',
    'claim_policy': 'inventory-existing-complete-sentence-literal-no-new-split-or-rewrite',
    'context_policy': 'source-premise-existing-original-R-sentence-hypothesis-Q-prompt-metadata-only',
    'projection_policy': 'unique-original-target-sentence-membership-no-character-max-or-smoothing',
    'risk_definition': '1-minus-max-chunk-entailment',
    'head_indices_entailment_contradiction_neutral': [0, 2, 1],
    'neutral_risk_mapping': 'non-entailment-through-one-minus-entailment',
    'unknown_risk': .5,
    'unknown_reasons': ['hypothesis_exhausts_pair_budget', 'empty_tokenized_hypothesis', 'empty_tokenized_source'],
    'fatal_errors_are_not_unknown_fallback': True, 'deduplicate_pairs': False,
    'planning_order': 'prepared-case-then-existing-sentence-then-source-window',
    'canary_policy': 'longest-eligible-framed-pair-first-fixed-order-tie',
    'canary_forward_calls': 2,
    'canary_comparison': 'same-FP32-HFEntailment-direct-vs-adapter-three-logits-bitwise',
    'canary_checks_independent_loader_equivalence': False, 'reuse_adapter_canary_once': True,
    'completed_forward_formula': 'K+1-if-K-positive-else-0-no-dedup',
    'cuda_allocator_cap_bytes': 6 * 1024**3,
    'minimum_available_host_bytes': 512 * 1024**2, 'minimum_free_gpu_bytes': 512 * 1024**2,
    'network_fallback': False, 'automatic_retry': False, 'target_label_fitting': False,
    'evaluation_after_all_scores_sealed': True,
    'evaluation_annotations_path': 'D:/projects/research/outputs/fusion-cache-20261102/prepared/test_Data2txt/annotations.jsonl',
    'evaluation_annotations_sha256': 'd6b2f44d12d68b520e5a28dd4d642b03a94854a2ec5b208a3ceee3f73be2b9d2',
    'evaluation_repetitions': 2000, 'evaluation_seed': 17, 'evaluation_label_granularity': 'native',
    'exposure_status': 'historically-exposed-exploratory-not-confirmatory',
    'old34case_results_reused_or_model_selection_metric': False,
}


def _protocol(path):
    value = data.read_json(path)
    # JSON equality alone would allow0=False and integer/floating repairs.
    if json.dumps(value, sort_keys=True) != json.dumps(DEFAULT_PROTOCOL, sort_keys=True):
        raise ValueError('The exact full-population NLI protocol is required')
    return value


def _seal(root):
    data.write_json(root / 'seal.json', {str(p.relative_to(root)): data.file_hash(p)
        for p in sorted(root.rglob('*')) if p.is_file() and p != root / 'seal.json'})


def _write_line(stream, row):
    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def _flush(stream):
    stream.flush()
    os.fsync(stream.fileno())


def _inputs(inputs, inventory, protocol):
    inputs, inventory, protocol = (Path(p).resolve() for p in (inputs, inventory, protocol))
    frozen = _protocol(protocol)
    expected = {inputs: frozen['original_inputs_sha256'],
                inputs.parent / 'manifest.json': frozen['original_manifest_sha256'],
                inventory: frozen['inventory_sha256']}
    if inputs.name != 'inputs.jsonl' or any(data.file_hash(p) != sha for p, sha in expected.items()):
        raise ValueError('Frozen original input, manifest or sentence inventory changed')
    manifest, inv = data.read_json(inputs.parent / 'manifest.json'), data.read_json(inventory)
    samples = list(read_samples(inputs))
    if (manifest['inputs_sha256'] != expected[inputs] or manifest['samples'] != len(samples)
            or len(samples) != frozen['expected_samples']
            or len({s.source_id for s in samples}) != frozen['expected_sources']
            or len(inv.get('records', [])) != len(samples)
            or inv.get('all900_original_axes_and_units_match') is not True):
        raise ValueError('The complete original population is required')
    for sample, record in zip(samples, inv['records'], strict=True):
        if (sample.task != frozen['task'] or sample.split != frozen['split']
                or sample.id != record['id'] or sample.source_id != record['source_id']
                or data.text_hash(sample.source) != record['source_sha256']
                or data.text_hash(sample.response) != record['response_sha256']):
            raise ValueError('Original sample/source/response identity changed')
    expected[protocol] = data.file_hash(protocol)
    return inputs, inventory, frozen, samples, inv['records'], {str(p): sha for p, sha in expected.items()}


def _backend_contract(backend, frozen):
    if (getattr(backend, 'execution_mode', None) != frozen['execution_mode']
            or nli._identity(backend.model_identity)['model_id'] != frozen['model_id']
            or backend.model_identity['revision'] != frozen['model_revision']):
        raise ValueError('The exact FP32 dataset backend mode/model is required')


def _response(sample, record):
    path = Path(record['primitive_response_path'])
    if data.file_hash(path) != record['primitive_response_sha256']:
        raise ValueError('Original primitive response metadata changed')
    response = data.read_json(path)
    if (set(response) != {'answer_ids', 'offsets', 'text', 'token_text', 'special_ids', 'units'}
            or response['text'] != sample.response
            or any(type(i) is not int or i < 0 for i in response['answer_ids'])
            or any(type(i) is not int or i < 0 for i in response['special_ids'])
            or len(response['offsets']) != len(response['answer_ids'])):
        raise ValueError('Original response token metadata is invalid')
    for span in response['offsets']:
        if (not isinstance(span, list) or len(span) != 2
                or any(type(i) is not int for i in span)
                or not 0 <= span[0] <= span[1] <= len(sample.response)):
            raise ValueError('Original response character offsets are invalid')
    cursor = 0
    scopes = {}
    for i, unit in enumerate(response['units']):
        a, b, left, right = (unit[k] for k in ('start', 'stop', 'text_start', 'text_stop'))
        if (any(type(x) is not int for x in (a, b, left, right))
                or a != cursor or not 0 <= left <= a < b <= right <= len(response['answer_ids'])):
            raise ValueError('Original capped measurement units do not cover the response')
        scopes.setdefault((left, right), []).append(i)
        cursor = b
    if cursor != len(response['answer_ids']):
        raise ValueError('Original measurement-unit suffix is missing')
    axis = answer_axis(response)
    owners, previous = [], 0
    if len(scopes) != len(record['claims']):
        raise ValueError('Original full-sentence scope count changed')
    for i, (((a, b), members), claim) in enumerate(zip(scopes.items(), record['claims'], strict=True)):
        targets = axis['target'][(axis['target'] >= a) & (axis['target'] < b)].tolist()
        lo = min((response['offsets'][t][0] for t in targets), default=None)
        hi = max((response['offsets'][t][1] for t in targets), default=None)
        literal = sample.response[lo:hi] if targets else ''
        if (a != previous or claim['claim_index'] != i or claim['original_sentence_token_span'] != [a, b]
                or claim['original_measurement_unit_ids'] != members or claim['target_indices'] != targets
                or claim['character_span'] != [lo, hi] or claim['claim_text'] != literal
                or claim['claim_sha256'] != data.text_hash(literal)):
            raise ValueError('Existing sentence scope, literal or unique target membership changed')
        owners.extend(targets)
        previous = b
    if owners != axis['target'].tolist() or previous != len(response['answer_ids']):
        raise ValueError('Every original target must have exactly one sentence owner')
    return response, axis


def _prepared(backend, source, claim):
    value = backend.prepare_pairs(source, claim)
    hyp, count, budget = value['hypothesis_tokens'], value['source_tokens'], value['budget']
    if (any(type(x) is not int or x < 0 for x in (hyp, count))
            or type(budget) is not int or budget != 509 - hyp
            or value['overlap_tokens'] != min(64, max(0, budget // 4))):
        raise ValueError('Existing NLI framing budget changed')
    reason = ('empty_tokenized_hypothesis' if hyp == 0 else 'hypothesis_exhausts_pair_budget' if budget <= 0
              else 'empty_tokenized_source' if count == 0 else None)
    if reason == 'empty_tokenized_hypothesis':
        return dict(value, status='unknown', reason=reason, chunks=[])
    if value['reason'] != reason or value['status'] != ('unknown' if reason else 'scored'):
        raise ValueError('NLI framing unknown state changed')
    cursor = 0
    for chunk in value['chunks']:
        nli._valid_pair(chunk)
        end = min(cursor + budget, count)
        if (chunk['start'] != cursor or chunk['end'] != end or not cursor < end
                or len(chunk['input_ids']) != end - cursor + hyp + 3):
            raise ValueError('NLI source windows omit tokens or alter pair framing')
        cursor = end - value['overlap_tokens'] if end < count else end
    if bool(value['chunks']) != (reason is None) or (value['chunks'] and cursor != count):
        raise ValueError('NLI source-window coverage is incomplete')
    return value


def prepare(inputs, inventory, protocol, output, backend):
    """Tokenizer-only complete planning; never calls backend.classify."""
    output = Path(output).resolve()
    inputs, inventory, frozen, samples, records, bindings = _inputs(inputs, inventory, protocol)
    _backend_contract(backend, frozen)
    if output.exists():
        raise FileExistsError('Retained NLI planning must not be overwritten')
    backend.verify_frozen_identity()
    identity, counts, longest = research_identity(), dict(units=0, pairs=0, unknown_units=0,
        valid_targets=0, measurement_units=0), None
    output.mkdir(parents=True)
    data.write_json(output / 'protocol.json', frozen)
    data.write_json(output / 'model_identity.json', backend.model_identity)
    data.write_json(output / 'source_identity.json', identity)
    with (output / 'units.jsonl').open('x', encoding='utf-8') as stream:
        for case_index, (sample, record) in enumerate(zip(samples, records, strict=True)):
            response, axis = _response(sample, record)
            counts['valid_targets'] += len(axis['target'])
            counts['measurement_units'] += len(response['units'])
            for claim in record['claims']:
                prepared = _prepared(backend, sample.source, claim['claim_text'])
                row = dict(case_index=case_index, id=sample.id, source_id=sample.source_id,
                           claim=claim, prepared=prepared)
                _write_line(stream, row)
                counts['units'] += 1
                counts['unknown_units'] += prepared['status'] == 'unknown'
                for chunk_index, chunk in enumerate(prepared['chunks']):
                    pair = {k: chunk[k] for k in ('input_ids', 'attention_mask')}
                    candidate = dict(case_index=case_index, id=sample.id,
                        claim_index=claim['claim_index'], chunk_index=chunk_index,
                        framed_tokens=len(pair['input_ids']), pair=pair)
                    if longest is None or candidate['framed_tokens'] > longest['framed_tokens']:
                        longest = candidate
                    counts['pairs'] += 1
            print(f'planned {case_index + 1}/{len(samples)} cases', flush=True) if (case_index + 1) % 100 == 0 else None
        _flush(stream)
    expected = {key: frozen[name] for key, name in (
        ('units', 'expected_sentence_units'), ('valid_targets', 'expected_original_valid_targets'),
        ('measurement_units', 'expected_measurement_units'))}
    if any(counts[k] != v for k, v in expected.items()):
        raise ValueError('The full original unit/token population changed')
    for path, digest in bindings.items():
        if data.file_hash(Path(path)) != digest:
            raise ValueError('Input identity changed during NLI planning')
    backend.verify_frozen_identity()
    completed_identity(identity)
    plan = dict(schema='local-nli-dataset-planning-v1', counts=counts, bindings=bindings,
        samples=len(samples), longest=longest, original_forward_budget=counts['pairs'] + bool(counts['pairs']),
        execution_mode=frozen['execution_mode'], model_identity=backend.model_identity,
        units_sha256=data.file_hash(output / 'units.jsonl'), annotation_reads=0, model_calls=0)
    data.write_json(output / 'plan.json', plan)
    _seal(output)
    return plan


def _raw(raw, pair):
    key = nli._digest(pair)
    values, costs = nli._batch(raw, [key], {key: pair}, 0)
    return values[key], costs


def _observe(backend, pair):
    """Return original backend data for retention BEFORE numerical validation."""
    return backend.classify([pair])


def predict(inputs, inventory, protocol, plan, output, backend):
    """Score a complete sealed plan; root must claim its K+1 budget first."""
    output, plan = Path(output).resolve(), Path(plan).resolve()
    inputs, inventory, frozen, samples, records, bindings = _inputs(inputs, inventory, protocol)
    _backend_contract(backend, frozen)
    data.verify_seal(plan, expected={'protocol.json', 'model_identity.json', 'source_identity.json', 'units.jsonl', 'plan.json'})
    planned = data.read_json(plan / 'plan.json')
    if (planned['bindings'] != bindings or data.read_json(plan / 'protocol.json') != frozen
            or planned['model_identity'] != backend.model_identity
            or planned['execution_mode'] != frozen['execution_mode']
            or planned['units_sha256'] != data.file_hash(plan / 'units.jsonl')
            or data.read_json(plan / 'source_identity.json')['source_sha256'] != research_identity()['source_sha256']):
        raise ValueError('NLI scoring plan/input/model/source identity changed')
    if output.exists():
        raise FileExistsError('Retained NLI scoring must not be overwritten or resumed')
    backend.verify_frozen_identity()
    source = research_identity()
    output.mkdir(parents=True)
    (output / 'predictions').mkdir()
    (output / 'units').mkdir()
    run = dict(input=str(inputs), input_sha256=bindings[str(inputs)],
        manifest_sha256=bindings[str(inputs.parent / 'manifest.json')], task=frozen['task'],
        method='fixed-external-deberta-nli', model_identity=backend.model_identity,
        protocol=frozen, protocol_sha256=bindings[str(Path(protocol).resolve())],
        inventory_sha256=bindings[str(inventory)], planning_root=str(plan),
        planning_seal_sha256=data.file_hash(plan / 'seal.json'), source_identity=source,
        original_forward_budget=planned['original_forward_budget'], annotation_reads=0)
    data.write_json(output / 'run.json', run)
    calls, costs = 0, dict(actual_forward_passes=0, input_tokens=0, padded_input_tokens=0, seconds=0.)
    unknown_tokens, unit_count, logical_pairs, reused, canary = 0, 0, 0, False, None
    total_targets, unknown_units, unknown_cases = 0, 0, 0

    def attempt(name):
        nonlocal calls
        calls += 1
        # Unique filenames preserve attempted calls even if classify is killed.
        data.write_json(output / f'attempt-{calls:06d}.json', dict(call=calls, name=name))

    def add_cost(value):
        for name in costs:
            costs[name] += value[name]

    try:
        if planned['longest'] is not None:
            selection = planned['longest']
            # Scoring may run in a fresh process. Reuse the existing tokenizer
            # initializer on this fixed original unit, without loading weights.
            sample = samples[selection['case_index']]
            claim = records[selection['case_index']]['claims'][selection['claim_index']]
            fresh = _prepared(backend, sample.source, claim['claim_text'])
            selected_unit = next((row for row in read_jsonl(plan / 'units.jsonl')
                if row['case_index'] == selection['case_index']
                and row['claim']['claim_index'] == selection['claim_index']), None)
            pair = selection['pair']
            if (selected_unit is None or selected_unit['prepared'] != fresh
                    or {key: fresh['chunks'][selection['chunk_index']][key]
                        for key in ('input_ids', 'attention_mask')} != pair):
                raise ValueError('Fresh scoring tokenizer differs from sealed fixed-canary framing')
            attempt('canary-direct')
            direct = backend.classify([pair])
            data.write_json(output / 'canary-direct-raw.json', dict(value=direct))
            direct_value, direct_cost = _raw(direct, pair)
            add_cost(direct_cost)
            attempt('canary-adapter')
            adapter_raw = _observe(backend, pair)
            data.write_json(output / 'canary-adapter-raw.json', dict(value=adapter_raw))
            adapter_value, adapter_cost = _raw(adapter_raw, pair)
            add_cost(adapter_cost)
            if (np.asarray(direct_value['logits'], dtype=np.float64).tobytes()
                    != np.asarray(adapter_value['logits'], dtype=np.float64).tobytes()
                    or direct_value['class_vector'] != adapter_value['class_vector']):
                raise ValueError('Same-FP32 direct/adapter canary logits differ')
            canary = (adapter_raw, adapter_value)
            data.write_json(output / 'canary.json', dict(passed=True, selection=planned['longest'],
                independent_loader_equivalence_checked=False, actual_calls=2))
        stream = iter(read_jsonl(plan / 'units.jsonl'))
        for case_index, (sample, record) in enumerate(zip(samples, records, strict=True)):
            response, axis = _response(sample, record)
            total_targets += len(axis['target'])
            risk = np.full(len(axis['target']), np.nan)
            unknown = np.zeros(len(axis['target']), dtype=bool)
            target_to_row = {int(t): i for i, t in enumerate(axis['target'])}
            case_dir = output / 'units' / data.prediction_path(Path('.'), sample.id).stem
            case_dir.mkdir()
            for claim in record['claims']:
                row = next(stream, None)
                if (row is None or row['case_index'] != case_index or row['id'] != sample.id
                        or row['source_id'] != sample.source_id or row['claim'] != claim):
                    raise ValueError('Planned sentence stream no longer matches original ownership')
                prepared, chunks = row['prepared'], []
                for chunk_index, chunk in enumerate(prepared['chunks']):
                    pair = {k: chunk[k] for k in ('input_ids', 'attention_mask')}
                    selection = planned['longest']
                    hit = (selection is not None and selection['case_index'] == case_index
                           and selection['claim_index'] == claim['claim_index'] and selection['chunk_index'] == chunk_index)
                    if hit:
                        if reused or pair != selection['pair'] or canary is None:
                            raise ValueError('Canary reuse identity changed')
                        raw, value = canary
                        reused = True
                    else:
                        attempt(f'{sample.id}/{claim["claim_index"]}/{chunk_index}')
                        raw = backend.classify([pair])
                        # Persist returned raw before numerical or cost validation.
                        data.write_json(case_dir / f'{claim["claim_index"]:05d}-{chunk_index:05d}-raw.json', dict(value=raw))
                        value, cost = _raw(raw, pair)
                        add_cost(cost)
                    chunks.append(dict(**chunk, raw=raw, class_vector=value['class_vector'], canary_reused=hit))
                    logical_pairs += 1
                if (prepared['status'] == 'unknown') != (not chunks):
                    raise ValueError('Planned unknown/chunk state changed')
                score = 1. - max(c['class_vector'][0] for c in chunks) if chunks else frozen['unknown_risk']
                selected = [target_to_row[t] for t in claim['target_indices']]
                if np.isfinite(risk[selected]).any():
                    raise ValueError('An original target belongs to more than one sentence')
                risk[selected], unknown[selected] = score, not chunks
                unit_count += 1
                unknown_units += not chunks
                data.write_json(case_dir / f'{claim["claim_index"]:05d}.json',
                    dict(claim=claim, prepared=prepared, chunks=chunks, risk=score,
                         unknown=not chunks, reason=prepared['reason']))
            if not np.isfinite(risk).all():
                raise ValueError('Some original targets have no finite risk')
            tokens = token_scores(axis, response, risk)
            prediction = dict(id=sample.id, source_id=sample.source_id,
                response_sha256=data.text_hash(sample.response), tokens=tokens,
                original_target_indices=axis['target'].tolist(), original_unit_indices=axis['unit_index'].tolist(),
                model_unknown_mask=unknown.tolist(), original_answer_ids=response['answer_ids'],
                original_offsets=response['offsets'])
            pipeline._validate_prediction(prediction, sample)
            data.write_json(data.prediction_path(output / 'predictions', sample.id), prediction)
            unknown_tokens += int(unknown.sum())
            unknown_cases += bool(unknown.any())
            print(f'scored {case_index + 1}/{len(samples)} cases; {calls} calls', flush=True)
        if next(stream, None) is not None:
            raise ValueError('Planned sentence stream contains extra units')
        if (logical_pairs != planned['counts']['pairs'] or unit_count != frozen['expected_sentence_units']
                or total_targets != frozen['expected_original_valid_targets']
                or unknown_units != planned['counts']['unknown_units']
                or calls != planned['original_forward_budget'] or costs['actual_forward_passes'] != calls
                or bool(canary) != reused):
            raise ValueError('Full-population NLI units/pairs/call accounting changed')
        for path, digest in bindings.items():
            if data.file_hash(Path(path)) != digest:
                raise ValueError('Frozen inference input changed')
        data.verify_seal(plan)
        backend.verify_frozen_identity()
        completed_identity(source)
        _seal(output / 'predictions')
        pipeline._verify_predictions(output / 'predictions')
        pipeline._complete_predictions(output, len(samples))
        result = dict(schema='local-nli-dataset-observation-complete-v1', samples=len(samples),
            sentence_units=unit_count, original_valid_targets=total_targets,
            logical_pairs=logical_pairs, costs=costs, actual_forward_attempts=calls,
            model_unknown_targets=unknown_tokens, model_unknown_units=unknown_units,
            model_unknown_cases=unknown_cases, unknown_risk=frozen['unknown_risk'],
            canary_reused=reused, annotation_reads=0, auroc=None)
        data.write_json(output / 'observation.json', result)
        _seal(output)
        return result
    except BaseException as error:
        data.write_json(output / 'failure.json', dict(type=type(error).__name__, message=str(error),
            actual_forward_attempts=calls, completed_costs=costs, completed_sentence_units=unit_count,
            partial_logical_pairs=logical_pairs, annotation_reads=0, completed=False))
        raise


def run(args):
    """Central CLI stage: --prepare-only, then scoring with an explicit --plan."""
    if getattr(args, 'reuse_failed', None):
        return recover(args)
    if getattr(args, 'device', 'cuda:0') != 'cuda:0':
        raise ValueError('The frozen NLI execution device is cuda:0')
    preparing = bool(getattr(args, 'prepare_only', False))
    if not preparing and not getattr(args, 'plan', None):
        raise ValueError('Scoring requires an explicit sealed --plan and root claim')
    identity = nli._identity(data.read_json(args.model_identity))
    backend = nli.HFEntailment(args.model, identity, admit_resources=not preparing,
                             execution_mode='dataset-fp32-batch1-v1')
    try:
        if preparing:
            return prepare(args.inputs, args.inventory, args.protocol, args.output, backend)
        return predict(args.inputs, args.inventory, args.protocol, args.plan, args.output, backend)
    finally:
        backend.close()


def _module_compatibility(old_path, new_path):
    """Require every old core AST node unchanged; only the dispatch may grow."""
    import ast
    old = ast.parse(Path(old_path).read_text(encoding='utf-8'))
    new = ast.parse(Path(new_path).read_text(encoding='utf-8'))
    if len(new.body) < len(old.body):
        raise ValueError('Recovery removed original collector source')
    for a, b in zip(old.body, new.body[:len(old.body)], strict=True):
        if isinstance(a, ast.FunctionDef) and a.name == 'run':
            candidate = list(b.body)
            if len(candidate) == len(a.body) + 1:
                gate = candidate.pop(1)
                expected = ast.parse("if getattr(args, 'reuse_failed', None):\n    return recover(args)").body[0]
                if ast.dump(gate) != ast.dump(expected):
                    raise ValueError('Unreviewed recovery dispatch changed')
            b = ast.FunctionDef(name=b.name, args=b.args, body=candidate,
                decorator_list=b.decorator_list, returns=b.returns, type_comment=b.type_comment,
                type_params=getattr(b, 'type_params', []))
        if ast.dump(a, include_attributes=False) != ast.dump(b, include_attributes=False):
            raise ValueError('Original collector core AST changed')
    if any(not isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in new.body[len(old.body):]):
        raise ValueError('Recovery may only append reviewed definitions')


def _recovery_source_proof(stage_source, previous):
    stage_source = Path(stage_source).resolve()
    current, old = research_identity(), previous['source_sha256']
    now = current['source_sha256']
    key = 'experiments/nli_dataset.py'
    if key not in old or data.file_hash(stage_source) != old[key] or set(old) != set(now):
        raise ValueError('Original installed collector source identity changed')
    changed = {name for name in old if old[name] != now[name]}
    if changed - {key, 'experiments/run.py'}:
        raise ValueError('A frozen numerical dependency or HF kernel changed')
    target = stage_source.parent.parent
    for relative, digest in old.items():
        path = (target / relative.replace('experiments.research/', 'experiments/research/')).resolve()
        if not path.is_relative_to(target) or data.file_hash(path) != digest:
            raise ValueError('Original installed dependency snapshot changed')
    for name in ('python', 'dependencies', 'threads', 'numpy_build'):
        if previous.get(name) != current.get(name):
            raise ValueError('Frozen numerical runtime changed during recovery')
    _module_compatibility(stage_source, Path(__file__))
    return dict(old_source_sha256=old, new_source_sha256=now,
        changed_only_dispatch_and_reviewed_recovery=sorted(changed), core_AST_unchanged=True,
        stage_source_path=str(stage_source), stage_source_sha256=data.file_hash(stage_source),
        HF_and_other_numerical_dependencies_byte_identical=True)


def _canonical_calls(plan_root):
    """Lazy physical call order, including original canaries and partial units."""
    plan_root = Path(plan_root)
    plan = data.read_json(plan_root / 'plan.json')
    selection = plan['longest']
    if selection is None:
        return
    for i, name in enumerate(('canary-direct', 'canary-adapter'), 1):
        yield dict(ordinal=i, name=name, pair=selection['pair'], raw=f'{name}-raw.json', logical_index=None)
    ordinal, logical = 2, 0
    for row in read_jsonl(plan_root / 'units.jsonl'):
        for chunk_index, chunk in enumerate(row['prepared']['chunks']):
            hit = (selection['case_index'] == row['case_index']
                and selection['claim_index'] == row['claim']['claim_index']
                and selection['chunk_index'] == chunk_index)
            if not hit:
                ordinal += 1
                stem = data.prediction_path(Path('.'), row['id']).stem
                name = f'{row["id"]}/{row["claim"]["claim_index"]}/{chunk_index}'
                raw = str(Path('units') / stem / f'{row["claim"]["claim_index"]:05d}-{chunk_index:05d}-raw.json')
                yield dict(ordinal=ordinal, name=name,
                    pair={k: chunk[k] for k in ('input_ids', 'attention_mask')}, raw=raw, logical_index=logical)
            logical += 1


def _recovery_prefix_proof(plan_root, failed_root):
    import hashlib
    failed_root, plan_root = Path(failed_root).resolve(), Path(plan_root).resolve()
    seal = data.verify_seal(failed_root)
    data.verify_seal(plan_root, expected={'protocol.json', 'model_identity.json', 'source_identity.json', 'units.jsonl', 'plan.json'})
    failure, run = data.read_json(failed_root / 'failure.json'), data.read_json(failed_root / 'run.json')
    plan = data.read_json(plan_root / 'plan.json')
    successful = failure['completed_costs']['actual_forward_passes']
    if (failure['type'] != 'RuntimeError' or failure['message'] != 'Dataset NLI requires the frozen512MiB host/GPU floors'
            or type(successful) is not int or successful < 2
            or failure['actual_forward_attempts'] != successful + 1 or failure['completed'] is not False
            or failure['annotation_reads'] != 0 or (failed_root / 'complete.json').exists()
            or run['planning_seal_sha256'] != data.file_hash(plan_root / 'seal.json')
            or run['original_forward_budget'] != plan['original_forward_budget']):
        raise ValueError('Only this sealed pre-forward resource refusal permits prefix recovery')
    costs = dict(actual_forward_passes=0, input_tokens=0, padded_input_tokens=0, seconds=0.)
    digest, raws, stream = hashlib.sha256(), set(), iter(_canonical_calls(plan_root))
    for ordinal in range(1, successful + 2):
        entry = next(stream, None)
        if entry is None or entry['ordinal'] != ordinal:
            raise ValueError('Canonical successful prefix has a hole')
        attempt = data.read_json(failed_root / f'attempt-{ordinal:06d}.json')
        if attempt != dict(call=ordinal, name=entry['name']):
            raise ValueError('Original request order or attempt identity changed')
        path = failed_root / entry['raw']
        if ordinal > successful:
            if path.exists() or entry['logical_index'] != failure['partial_logical_pairs']:
                raise ValueError('Refused request has raw model output or a different prefix')
            break
        if entry['raw'] not in seal or data.file_hash(path) != seal[entry['raw']]:
            raise ValueError('A successful retained raw is missing or changed')
        raw = data.read_json(path)['value']
        _, cost = _raw(raw, entry['pair'])
        for key in costs:
            costs[key] += cost[key]
        raws.add(entry['raw'])
        digest.update(json.dumps(dict(ordinal=ordinal, name=entry['name'], pair=entry['pair'],
            raw_sha256=seal[entry['raw']]), sort_keys=True, separators=(',', ':')).encode('utf-8'))
    if (costs != failure['completed_costs']
            or {name for name in seal if name.endswith('-raw.json')} != raws
            or {p.name for p in failed_root.glob('attempt-*.json')}
                != {f'attempt-{i:06d}.json' for i in range(1, successful + 2)}):
        raise ValueError('Successful physical raw/cost/attempt prefix is incomplete')
    canary = data.read_json(failed_root / 'canary.json')
    if (canary['passed'] is not True or canary['actual_calls'] != 2
            or canary['selection'] != plan['longest']):
        raise ValueError('Original valid interface qualification is missing')
    return dict(successful_model_results=successful, previous_guard_refusals=1,
        previous_interface_attempts=successful + 1, prefix_sha256=digest.hexdigest(),
        old_costs=costs, remaining_model_calls=plan['original_forward_budget'] - successful,
        old_plan=str(plan_root), failed_root=str(failed_root),
        old_plan_seal_sha256=data.file_hash(plan_root / 'seal.json'),
        failed_seal_sha256=data.file_hash(failed_root / 'seal.json'),
        failed_run_source=run['source_identity'])


class ReplayPrefixBackend:
    """Return the exact retained physical prefix before any fresh model call."""
    def __init__(self, base, old_plan, failed_root):
        self.base, self.old_plan, self.failed_root = base, Path(old_plan), Path(failed_root)
        self.proof = _recovery_prefix_proof(self.old_plan, self.failed_root)
        self._raw_sha256 = {name: digest for name, digest in data.read_json(self.failed_root / 'seal.json').items()
                           if name.endswith('-raw.json')}
        self.model_identity, self.execution_mode = base.model_identity, base.execution_mode
        self._calls, self.cached_calls, self.new_model_calls = iter(_canonical_calls(self.old_plan)), 0, 0
        self.interface_requests = 0
        self.new_interface_requests = 0

    def verify_frozen_identity(self):
        return self.base.verify_frozen_identity()

    def prepare_pairs(self, source, hypothesis):
        return self.base.prepare_pairs(source, hypothesis)

    def classify(self, pairs):
        if len(pairs) != 1:
            raise ValueError('Recovery preserves the strict batch1 contract')
        nli._valid_pair(pairs[0])
        entry = next(self._calls, None)
        self.interface_requests += 1
        if (entry is None or entry['ordinal'] != self.interface_requests
                or entry['pair'] != pairs[0]):
            raise ValueError('Recovery request differs from the fixed canonical call sequence')
        if self.interface_requests <= self.proof['successful_model_results']:
            path = self.failed_root / entry['raw']
            if data.file_hash(path) != self._raw_sha256[entry['raw']]:
                raise ValueError('Retained raw changed before replay')
            raw = data.read_json(path)['value']
            _raw(raw, pairs[0])
            self.cached_calls += 1
            return raw
        self.new_interface_requests += 1
        raw = self.base.classify(pairs)
        self.new_model_calls += 1
        return raw

    def close(self):
        return self.base.close()


def prepare_recovery(inputs, inventory, protocol, old_plan, failed, stage_source, output, backend):
    """Derive a compatible plan without tokenization, weights or forward calls."""
    import shutil
    output, old_plan = Path(output).resolve(), Path(old_plan).resolve()
    _, _, frozen, _, _, bindings = _inputs(inputs, inventory, protocol)
    _backend_contract(backend, frozen)
    backend.verify_frozen_identity()
    if output.exists():
        raise FileExistsError('Retained recovery planning must not be overwritten')
    proof = _recovery_prefix_proof(old_plan, failed)
    source_proof = _recovery_source_proof(stage_source, proof.pop('failed_run_source'))
    planned = data.read_json(old_plan / 'plan.json')
    if (planned['bindings'] != bindings or planned['model_identity'] != backend.model_identity
            or data.read_json(old_plan / 'protocol.json') != frozen):
        raise ValueError('Recovery must retain the complete original population/protocol/model')
    output.mkdir(parents=True)
    for name in ('units.jsonl', 'protocol.json', 'model_identity.json'):
        shutil.copyfile(old_plan / name, output / name)
    data.write_json(output / 'source_identity.json', research_identity())
    planned['recovery'] = dict(proof, source_proof=source_proof,
        explicit_new_claim_not_automatic_retry=True, fresh_resident_canary_performed=False)
    data.write_json(output / 'plan.json', planned)
    _seal(output)
    return planned


def recover(args):
    """Explicit reviewed continuation; preserve the original predict AST/math."""
    if (not getattr(args, 'plan', None) or not getattr(args, 'reuse_stage_source', None)
            or getattr(args, 'device', 'cuda:0') != 'cuda:0'):
        raise ValueError('Recovery needs old stage source, a sealed plan and the frozen CUDA device')
    preparing = bool(getattr(args, 'prepare_only', False))
    identity = nli._identity(data.read_json(args.model_identity))
    base = nli.HFEntailment(args.model, identity, admit_resources=not preparing,
        execution_mode='dataset-fp32-batch1-v1')
    if preparing:
        try:
            return prepare_recovery(args.inputs, args.inventory, args.protocol, args.plan,
                args.reuse_failed, args.reuse_stage_source, args.output, base)
        finally:
            base.close()
    planned = data.read_json(Path(args.plan) / 'plan.json')
    recovery = planned['recovery']
    if (Path(args.reuse_failed).resolve() != Path(recovery['failed_root']).resolve()
            or data.file_hash(Path(args.reuse_failed) / 'seal.json') != recovery['failed_seal_sha256']):
        base.close()
        raise ValueError('A different failed archive cannot supply recovery evidence')
    previous = data.read_json(Path(args.reuse_failed) / 'run.json')['source_identity']
    if _recovery_source_proof(args.reuse_stage_source, previous) != recovery['source_proof']:
        base.close()
        raise ValueError('Recovery source compatibility changed')
    proxy = ReplayPrefixBackend(base, recovery['old_plan'], args.reuse_failed)
    result, completed = None, False
    try:
        result = predict(args.inputs, args.inventory, args.protocol, args.plan, args.output, proxy)
        if (proxy.cached_calls != recovery['successful_model_results']
                or proxy.new_model_calls != recovery['remaining_model_calls']):
            raise ValueError('Recovery cache/new physical call accounting differs')
        completed = True
        return result
    finally:
        base.close()
        output = Path(args.output).resolve()
        if output.exists():
            if not (output / 'seal.json').exists():
                _seal(output)
            data.write_json(output.with_name(output.name + '.recovery.json'), dict(
                completed=completed, old_failed_seal_sha256=recovery['failed_seal_sha256'],
                compatible_plan_seal_sha256=data.file_hash(Path(args.plan) / 'seal.json'),
                output_seal_sha256=data.file_hash(output / 'seal.json'),
                replayed_valid_model_results=proxy.cached_calls, new_successful_model_calls=proxy.new_model_calls,
                current_proxy_interface_requests=proxy.interface_requests,
                previous_successful_model_calls=recovery['successful_model_results'],
                previous_guard_refusals=recovery['previous_guard_refusals'],
                previous_physical_interface_requests=recovery['previous_interface_attempts'],
                new_physical_interface_requests=proxy.new_interface_requests,
                new_requests_without_returned_raw=proxy.new_interface_requests - proxy.new_model_calls,
                combined_successful_model_calls=recovery['successful_model_results'] + proxy.new_model_calls,
                combined_physical_interface_requests=recovery['previous_interface_attempts']
                    + proxy.new_interface_requests,
                predict_attempt_and_cost_scope='merged canonical evidence, including replay; not current-process physical calls',
                cached_canaries_reuse_original_qualification=True, fresh_resident_canary_performed=False,
                automatic_retry=False, target_annotation_reads=0))
