"""Posthoc exact-cache proof only: strip suffix, never call a model or reducer."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
INSTALLED = HERE / 'installed'
OUT = HERE / 'context-stripping-reuse.json'
MARKDOWN = HERE / 'context-stripping-reuse.md'
sys.dont_write_bytecode = True
sys.path.insert(0, str(INSTALLED))
from experiments import data, verdict_qualification as shared
from experiments import semantic_qualification as semantic
from experiments import datasets
from experiments.research import focused_verdict as focus, semantic_verdict as verdict


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    require('construction' not in path.parts and path.name not in
            {'construction_truth.jsonl', 'evaluation_only.jsonl', 'annotations.jsonl'},
            'This proof must not parse truth files')
    return data.read_json(path)


def main():
    require(not OUT.exists() and not MARKDOWN.exists(), 'Existing proof is retained; do not rerun or overwrite')
    observer, child = HERE / 'burden_v1', HERE / 'burden_v1/observations'
    initial_seal_hash = data.file_hash(observer / 'seal.json')
    data.verify_seal(observer)
    plan, child_plan = read(observer / 'plan.json'), read(child / 'plan.json')
    source = read(observer / 'source_identity.json')
    helpers = {'experiments/data.py': data, 'experiments/datasets.py': datasets,
        'experiments/verdict_qualification.py': shared, 'experiments/semantic_qualification.py': semantic,
        'experiments.research/focused_verdict.py': focus, 'experiments.research/semantic_verdict.py': verdict}
    helper_hashes = {}
    for name, module in helpers.items():
        path = Path(module.__file__).resolve()
        require(path.is_relative_to(INSTALLED), 'A helper import escaped immutable109 target')
        helper_hashes[name] = data.file_hash(path)
        require(helper_hashes[name] == source['source_sha256'][name], 'Archived helper identity differs: ' + name)
    inputs = HERE / 'inference/inputs.jsonl'
    design = read(HERE / 'inference/plan.json')
    require(data.file_hash(inputs) == plan['original_inputs_sha256'], 'Frozen inference input changed')
    samples = list(datasets.read_samples(inputs))
    metadata = design['records']
    require(len(samples) == len(metadata) == 48 and [s.id for s in samples] == [r['id'] for r in metadata],
            'The full original48 input/metadata order is required')
    by_id = {sample.id: sample for sample in samples}
    def group(record):
        return tuple(record['factor_cell'][k] for k in
            ('relation_family', 'vocabulary', 'source_direction', 'response_direction'))
    bases = {group(r): r for r in metadata if r['factor_cell']['distractor_count'] == 0}
    require(len(bases) == 16, 'Expected exactly16 zero-burden original cases')
    flats = {(r['original_id'], r['mode']): r for r in plan['flat_records']}
    child_rows = {r['id']: r for r in child_plan['records']}
    completion = read(observer / 'complete.json')
    parent_rows = {r['id']: (i, r) for i, r in enumerate(completion['records'])}
    child_completion = read(child / 'complete.json')
    case_files = {r['id']: r for r in child_completion['records']}
    model = read(observer / 'model_identity.json')
    require(shared._digest(model) == child_plan['model_identity_sha256']
        and shared._digest(source) == child_plan['source_identity_sha256'], 'Child model/source identity changed')
    cache = {}
    def execution(key):
        if key not in cache:
            root = child / 'requests' / key
            data.verify_seal(root)
            value = read(root / 'request.json')
            require(shared._digest(value) == key, 'Request dictionary/key mismatch')
            require(value['model_identity'] == model and value['protocol_sha256'] == child_plan['protocol_sha256']
                and value['source_identity_sha256'] == shared._digest(source), 'Execution identities changed')
            cache[key] = value
        return cache[key]
    mappings, risk_refs, unique, counts = [], {}, set(), Counter()
    for record, sample in zip(metadata, samples, strict=True):
        base_record = bases[group(record)]
        base_sample = by_id[base_record['id']]
        context, transform = shared._context(sample.inputs)
        base_context, base_transform = shared._context(base_sample.inputs)
        require(sample.source == base_sample.source and sample.prompt == base_sample.prompt
            and context == base_context and transform == base_transform, 'Source/prompt/context differs from burden0')
        units = focus.response_units(sample.response)
        require(bool(units), 'Missing deterministic target unit')
        stripped = units[0]['text']
        require(stripped == base_sample.response and units[0]['start'] == 0,
                'Suffix stripping does not reproduce the complete burden0 response')
        target = flats[sample.id, 'target_first_unit']
        base_target = flats[base_sample.id, 'target_first_unit']
        require(target['focus_span'] == [0, len(stripped)] and target['focus_text'] == stripped,
                'Original first-unit focus span/text changed')
        base_row = child_rows[base_target['flat_id']]
        requests = child_rows[target['flat_id']]['requests']
        require(len(requests) == len(base_row['requests']) == 6, 'Every case must retain all6 assignments')
        for item, base_item in zip(requests, base_row['requests'], strict=True):
            assignment = item['label_assignment']
            require(assignment == base_item['label_assignment'], 'Label-assignment order changed')
            original = execution(item['request_key'])
            expected_original = focus.build_request(sample.source, sample.response, context,
                target['focus_span'], assignment)['messages']
            require(original['messages'] == expected_original, 'Archived original full-context request changed')
            candidate = deepcopy(original)
            candidate['messages'] = focus.build_request(sample.source, stripped, context,
                [0, len(stripped)], assignment)['messages']
            key = shared._digest(candidate)
            existing = execution(base_item['request_key'])
            require(candidate == existing and key == base_item['request_key'],
                    'Would-be stripped execution is not exactly an existing burden0 execution')
            unique.add(key); counts[key] += 1
            mappings.append(dict(original_id=sample.id, burden=record['factor_cell']['distractor_count'],
                label_assignment=assignment, original_target_request_key=item['request_key'],
                would_be_request_key=key, reused_zero_burden_request_key=base_item['request_key'],
                zero_burden_original_id=base_sample.id, zero_burden_target_flat_id=base_target['flat_id'],
                exact_execution_dictionary_equal=True, source_prompt_context_unchanged=True))
        if base_sample.id not in risk_refs:
            index, row = parent_rows[base_sample.id]
            require(row['whole']['risk'] == row['units'][0]['risk']
                and row['whole']['class_scores'] == row['units'][0]['class_scores'], 'Burden0 whole/target readouts differ')
            child_case = case_files[base_target['flat_id']]
            risk_refs[base_sample.id] = dict(parent_complete_json_pointer=f'/records/{index}/units/0',
                zero_burden_target_flat_id=base_target['flat_id'], child_case_file=child_case['file'],
                child_case_sha256=child_case['sha256'], risk=row['units'][0]['risk'],
                unknown_fallback=row['units'][0]['unknown_fallback'], status=row['units'][0]['status'],
                risk_is_copied_from_sealed_b0_readout=True, whole_alias_same_risk=True)
    expected_keys = {item['request_key'] for r in bases.values()
                     for item in child_rows[flats[r['id'], 'target_first_unit']['flat_id']]['requests']}
    require(len(mappings) == 288 and len(unique) == len(expected_keys) == 96 and unique == expected_keys
        and set(counts.values()) == {3}, 'Logical288 / existing96 / threefold reuse proof failed')
    review = HERE / 'posthoc_review_v1'
    review_seal = read(review / 'seal.json')
    require(data.file_hash(review / 'summary.json') == review_seal['summary.json'], 'Released summary changed')
    summary = read(review / 'summary.json')
    b0 = {m: summary['by_mode_and_burden'][m]['0'] for m in ('whole', 'target_first_unit')}
    require(b0['whole'] == b0['target_first_unit'], 'The sealed burden0 statistics are not identical')
    require(data.file_hash(observer / 'seal.json') == initial_seal_hash, 'Observer resealed during proof')
    bindings = [inputs, HERE / 'inference/plan.json', HERE / 'inference/seal.json',
        observer / 'plan.json', observer / 'complete.json', observer / 'seal.json',
        child / 'plan.json', child / 'protocol.json', child / 'complete.json', child / 'seal.json',
        observer / 'model_identity.json', observer / 'source_identity.json', review / 'summary.json', review / 'seal.json']
    result = dict(schema='h007-posthoc-context-stripping-cache-reuse-v1', status='exact_reuse_proved',
        created_utc=datetime.now(timezone.utc).isoformat(), posthoc_diagnostic=True, new_model_calls=0,
        new_tokenizer_calls=0, new_scores_or_metrics_computed=False, parameter_selection=False,
        formal_truth_parsed=False, candidate_logical_requests=288, unique_candidate_executions=96,
        unique_existing_zero_burden_executions=96, all_candidate_keys_already_cached=True,
        cache_key_reuse_multiplicity=3, zero_burden_original_cases=16, original_cases_retained=48,
        immutable_helper_hashes=helper_hashes, artifact_sha256={str(p.relative_to(HERE)):data.file_hash(p) for p in bindings},
        risk_references=risk_refs, mappings=mappings, unchanged_b0_statistics_copied_from_summary=b0,
        interpretation='Removing all answer suffix context exactly reuses the already observed burden0 instrument; it is not a new model experiment or a repair of the existing minimal relation failures.')
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')
    metric = b0['target_first_unit']
    text = f'''# H007 后缀去除：精确缓存复用证明

结论：48答×6分配=288个 would-be target 请求，逐个完整 execution 字典都与现有零负担16答×6=96个独立请求完全相同；每个已有 key 被引用3次。不存在新模型执行，不能把删除回答后缀重新包装成新实验。

操作仅将原回答改为首个自动精确单元，保留其固定尾空格，并删除所有后续字符。来源、原 prompt、来源占位转换后的 context、标签分配、model identity、protocol/source hash、scoring 和 admission 字段都保持；任何一项不等都会立即失败。实际使用不可修改109文件目标中的 focus.build_request/shared._context；助手未改 core/state。

风险只引用16条已封存 b0 parent completion 及其 child case 路径/哈希，没有重新聚合 logits、重算风险或读取正式 truth。48条逻辑实例仍全部列在 JSON 的288条映射中；该重复清单不是新独立观测。

直接引用 sealed summary 的 b0 数值：正确配对方向 {metric['correctly_ordered_pairs']}/{metric['paired_cells']}，平局 {metric['tied_pairs']}；mean_wrong_minus_supported_risk={metric['mean_wrong_minus_supported_risk']!r}，AUROC={metric['auroc']!r}，AP={metric['ap']!r}。whole 与 target 的这些既有值完全相同，去后缀不会改变它们，也不能修复最小关系条件已有的错误。

new_model_calls=0；posthoc_diagnostic=True；new_scores_or_metrics_computed=False。完整 input/design/observer/child-plan/summary/seal 绑定、96个对应缓存 key、16个风险引用与288条逐分配映射见 context-stripping-reuse.json。观察封存 SHA256：`{initial_seal_hash}`。
'''
    with MARKDOWN.open('x', encoding='utf-8') as stream:
        stream.write(text)
    print(json.dumps({k: result[k] for k in ('status', 'candidate_logical_requests',
        'unique_candidate_executions', 'unique_existing_zero_burden_executions', 'new_model_calls')}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status='proof_failed_no_model_call', error_type=type(error).__name__,
            error=str(error), new_model_calls=0, traceback=traceback.format_exc()), ensure_ascii=False))
        raise SystemExit(1)
