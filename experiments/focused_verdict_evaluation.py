"""CPU-only review of the complete sealed matched whole/all-unit focus pilot."""
from collections import Counter
import json
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score

from . import data, focused_verdict_qualification as focused
from . import verdict_evaluation as reader, typed_verdict_evaluation as oldreader
from . import verdict_qualification as shared
from .datasets import Sample, read_samples
from .provenance import completed_identity, research_identity
from .research import focused_verdict as focus


DEFAULT_PROTOCOL = dict(schema='focused-verdict-evaluation-v1', purpose='posthoc_exploratory_evaluation',
    required_cases=34, natural_cases=18, constructed_cases=16, modes=['whole','unit'],
    permutations=6, whole_case_population=True, model_calls_allowed=False,
    tokenizer_calls_allowed=False, target_label_selection_allowed=False,
    threshold_sweeps_allowed=False, fitted_calibration_allowed=False,
    unknown_risk=1.0, natural_axis='released_token_span_character_axis_or_native_sentence_axis',
    sentence_reduction='maximum_overlapping_unit_risk',
    controls_reduction='maximum_all_unit_risks_empty_response_zero',
    token_label_rewrites_allowed=False)
_HELPERS = ('experiments/focused_verdict_qualification.py',
    'experiments.research/focused_verdict.py', 'experiments/verdict_qualification.py',
    'experiments.research/semantic_verdict.py', 'experiments/semantic_qualification.py')
_PANEL_FILES = {'inputs.jsonl','manifest.json','evaluation_only.jsonl'}
_require = oldreader._require


def _inventory(root):
    """Admit the entire nested member tree before any hash/source identity call."""
    names = {'plan.json','complete.json','protocol.json','source_identity.json',
             'model_identity.json','shared_protocol.json','prepared_flat','observations','seal.json'}
    _require(root.resolve() == root and root.is_dir() and {p.name for p in root.iterdir()} == names,
             'Incomplete or aliased focused archive')
    members = set()
    for name in names-{'prepared_flat','observations'}:
        reader._member(root,name)
        if name != 'seal.json': members.add(name)
    prepared = reader._member(root,'prepared_flat',directory=True)
    _require({p.name for p in prepared.iterdir()} == {'inputs.jsonl','seal.json'},
             'Incomplete prepared focus inputs')
    for name in ('inputs.jsonl','seal.json'): reader._member(prepared,name)
    prepared_seal,prepared_bytes = reader._seal_content(prepared,{'inputs.jsonl'})
    members.update(str((prepared/name).relative_to(root)) for name in ('inputs.jsonl','seal.json'))
    child = reader._member(root,'observations',directory=True)
    child_members,child_seal,child_bytes,grandchildren = reader._observation_inventory(child)
    members.update(str((child/name).relative_to(root)) for name in child_members|{'seal.json'})
    seal,raw = reader._seal_content(root,members)
    return dict(members=members,seal=seal,hash=data.text_hash(raw.decode('utf-8')),
        prepared=(prepared,prepared_seal,data.text_hash(prepared_bytes.decode('utf-8'))),
        child=(child,child_members,child_seal,data.text_hash(child_bytes.decode('utf-8')),grandchildren))


def _snapshot(root,inventory):
    reader._verify_observation_snapshot(root,inventory['members'],inventory['seal'],inventory['hash'],[])
    prepared,seal,digest = inventory['prepared']
    reader._verify_observation_snapshot(prepared,{'inputs.jsonl'},seal,digest,[])
    reader._verify_observation_snapshot(*inventory['child'])


def _identity(sample):
    return dict(id=sample.id,source_id=sample.source_id,split=sample.split,task=sample.task,
                generator=sample.generator,evidence_path=sample.evidence_path,inputs=sample.inputs)


def _validate(inputs,root,current):
    plan = data.read_json(root/'plan.json')
    protocol = focused._protocol(root/'protocol.json')
    model = shared._model_identity(data.read_json(root/'model_identity.json'))
    source = data.read_json(root/'source_identity.json')
    for name in _HELPERS:
        digest = source.get('source_sha256',{}).get(name)
        _require(reader._sha(digest) and digest == current.get('source_sha256',{}).get(name),
                 'Incompatible archived observation helper: '+name)
    input_hash = data.file_hash(inputs)
    _require(input_hash == protocol['original_inputs_sha256'], 'Original input hash changed')
    samples = list(read_samples(inputs))
    _require(len(samples) == 34 and len({s.id for s in samples}) == 34, 'Whole original 34-case population required')
    originals,flat_rows,flat_records = focused._prepare(samples)
    flat_bytes = ''.join(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
                         for row in flat_rows).encode('utf-8')
    flat_inputs = root/'prepared_flat/inputs.jsonl'
    _require(flat_inputs.read_bytes() == flat_bytes, 'Prepared full input strings/order changed')
    flat_hash = data.file_hash(flat_inputs)
    child_protocol = dict(protocol,expected_cases=len(flat_rows),max_forward_passes=6*len(flat_rows))
    reader._equal(data.read_json(root/'shared_protocol.json'),child_protocol,'Shared protocol changed')
    expected = dict(schema='focused-verdict-qualification-plan-v1',state='frozen_before_observation',
        original_inputs_path=str(inputs),original_inputs_sha256=input_hash,protocol_sha256=plan.get('protocol_sha256'),
        protocol=protocol,source_identity=source,model_identity=model,prepared_inputs_sha256=flat_hash,
        prepared_cases=len(flat_rows),logical_requests=6*len(flat_rows),max_forward_passes=6*len(flat_rows),
        records=originals,flat_records=flat_records,evaluation_material_accessed=False,metadata_ids_are_model_inputs=False,
        matched_whole_is_new_necessary_granularity_control=True,h003_observations_imported=False,
        limitations=list(focus.LIMITATIONS))
    _require(reader._sha(plan.get('protocol_sha256')), 'Missing original protocol hash')
    reader._equal(plan,expected,'Original focus plan/partition/identity changed')
    child = root/'observations'
    child_hash = data.file_hash(root/'shared_protocol.json')
    reader._equal(data.read_json(child/'protocol.json'),child_protocol,'Child protocol changed')
    reader._equal(data.read_json(child/'source_identity.json'),source,'Child source changed')
    reader._equal(data.read_json(child/'model_identity.json'),model,'Child model changed')
    rows,executions = [],{}
    flat_samples = [Sample(**row) for row in flat_rows]
    for sample,metadata in zip(flat_samples,flat_records,strict=True):
        context,transformation = shared._context(sample.inputs)
        requests = []
        for assignment in focus.label_assignments():
            request = focus.build_request(sample.source,sample.response,context,metadata['focus_span'],assignment)
            execution = dict(messages=request['messages'],labels=list(focus.LABELS),model_identity=model,
                protocol_sha256=child_hash,source_identity_sha256=shared._digest(source),
                scoring=dict(use_cache=False,logits_to_keep=1,full_prefill=True,output_tokens=0),
                admission=dict(max_input_tokens=child_protocol['max_input_tokens']))
            key = shared._digest(execution)
            executions[key] = execution
            requests.append(dict(request_key=key,label_assignment=assignment))
        rows.append(dict(id=sample.id,source_id=sample.source_id,
            input_text_sha256={k:data.text_hash(v) for k,v in sample.inputs.items()},
            pre_generation_context_sha256=data.text_hash(context),context_transformation=transformation,requests=requests))
    child_plan = dict(schema='verdict-qualification-plan-v1',state='observations_only',
        inputs_path=str(flat_inputs),inputs_sha256=flat_hash,protocol_sha256=child_hash,protocol=child_protocol,
        model_identity_sha256=shared._digest(model),source_identity_sha256=shared._digest(source),records=rows,
        evaluation_material_accessed=False,case_ids_are_model_inputs=False,engineering_failure_is_semantic_refutation=False)
    reader._equal(data.read_json(child/'plan.json'),child_plan,'Child full request roster changed')
    _require({p.name for p in (child/'requests').iterdir()} == set(executions)
        and {p.name for p in (child/'cases').iterdir()} == {data.text_hash(s.id) for s in flat_samples},
        'Incomplete full focus request/case population')
    responses = {key:shared._cached(child/'requests',key,value) for key,value in executions.items()}
    _require(all(value is not None for value in responses.values()),'Missing retained request')
    cases,records = {},[]
    for sample,row in zip(flat_samples,rows,strict=True):
        case = reader._case(sample,row,child,responses,child_protocol)
        cases[sample.id] = case
        path = child/'cases'/data.text_hash(sample.id)/'case.json'
        records.append(dict(id=sample.id,file=str(path.relative_to(child)),sha256=data.file_hash(path)))
    completion = reader._completion(list(cases.values()),responses,records,child_protocol)
    reader._equal(data.read_json(child/'complete.json'),completion,'Child costs/coverage/completion changed')
    readouts = []
    for original in originals:
        readouts.append(dict(id=original['id'],source_id=original['source_id'],
            whole=dict(flat_id=original['whole_focus_id'],**focused._risk(cases[original['whole_focus_id']],1.0)),
            units=[dict(unit,**focused._risk(cases[unit['flat_id']],1.0)) for unit in original['units']]))
    complete = dict(schema='focused-verdict-qualification-complete-v1',state='observations_sealed_not_evaluated',
        coverage=dict(original_cases=34,whole_focuses=34,unit_focuses=len(flat_rows)-34,flat_cases=len(flat_rows),
            scored_focuses=completion['coverage']['scored_cases'],unknown_focuses=completion['coverage']['unknown_cases'],
            unknown_units=sum(u['unknown_fallback'] for r in readouts for u in r['units']),
            empty_responses=sum(not s.response for s in samples)),costs=completion['costs'],records=readouts,
        conservative_unknown_risk=1.0,source_identity_sha256=shared._digest(source),
        model_identity_sha256=shared._digest(model),evaluation_material_accessed=False,
        token_projection_performed=False,auroc_computed=False,h003_observations_imported=False)
    reader._equal(data.read_json(root/'complete.json'),complete,'Parent risk/cost/coverage completion changed')
    return samples,readouts,cases,plan,complete


def _metrics(labels,scores,unknown):
    known = [(y,s) for y,s in zip(labels,scores,strict=True) if y is not None]
    y,s = [r[0] for r in known],[r[1] for r in known]
    positives = sum(y)
    return dict(evaluation_units=len(labels),known_units=len(y),known_positive=positives,
        known_negative=len(y)-positives,missing_label_units=len(labels)-len(y),
        unknown_score_units=sum(unknown),unknown_score_known_label_units=sum(u and label is not None for u,label in zip(unknown,labels,strict=True)),
        auroc=float(roc_auc_score(y,s)) if 0 < positives < len(y) else None,
        ap=float(average_precision_score(y,s)) if positives else None)


def _natural(sample,readout,released):
    annotation = released['released_annotation_record']
    spans = annotation['spans']
    units = [{k:u[k] for k in ('start','end','text')} for u in readout['units']]
    if annotation['granularity'] == 'sentence':
        offsets = [(span['start'],span['end']) for span in spans]
        labels = [None if annotation['coverage'] == 'unalignable' else span['label'] for span in spans]
        axis,denominator = 'native_released_sentence_spans','sentence'
    else:
        offsets = [(index,index+1) for index in range(len(sample.response))]
        labels = []
        for start,end in offsets:
            overlapping = [span['label'] for span in spans if start < span['end'] and end > span['start']]
            labels.append(None if annotation['coverage'] == 'unalignable' else
                1 if 1 in overlapping else None if None in overlapping else
                0 if 0 in overlapping or annotation['coverage'] == 'complete' else None)
        axis,denominator = 'original_unicode_characters_from_released_token_spans','character'
    projected = focus.project_token_risks(units,[None if u['unknown_fallback'] else u['risk'] for u in readout['units']],offsets)
    covered_characters = sum(any(span['start'] <= i < span['end'] for span in spans)
                             for i in range(len(sample.response)))
    return dict(granularity=annotation['granularity'],annotation_coverage=annotation['coverage'],
        axis=axis,denominator_granularity=denominator,native_annotation_unchanged=True,
        native_units=[dict(span) for span in spans],evaluation_offsets=offsets,labels=labels,
        whole=dict(risks=[readout['whole']['risk']]*len(offsets),unknown_mask=[readout['whole']['unknown_fallback']]*len(offsets)),
        unit=projected,unannotated_characters_negative_only_with_complete_token_coverage=True,
        bpe_token_metrics_created=False,
        released_span_covered_characters=covered_characters,
        outside_released_span_characters=len(sample.response)-covered_characters,
        source_response_characters=len(sample.response))


def _natural_summary(packets):
    groups = {}
    for packet in packets:
        annotation = packet['native_evaluation']
        cohort = packet['released_evaluation_record']['cohort']
        key = cohort+'::'+annotation['granularity']
        group = groups.setdefault(key,dict(cohort=cohort,granularity=annotation['granularity'],
            cases=0,coverage=Counter(),labels=[],whole=[],unit=[],whole_unknown=[],unit_unknown=[],
            response_characters=0,released_span_covered_characters=0,outside_released_span_characters=0))
        group['cases'] += 1
        group['coverage'][annotation['annotation_coverage']] += 1
        for name in ('released_span_covered_characters','outside_released_span_characters'):
            group[name] += annotation[name]
        group['response_characters'] += annotation['source_response_characters']
        group['labels'].extend(annotation['labels'])
        for mode in ('whole','unit'):
            group[mode].extend(annotation[mode]['risks'])
            group[mode+'_unknown'].extend(annotation[mode]['unknown_mask'])
    result = {}
    for key,g in groups.items():
        denominator = 'character' if g['granularity'] == 'token' else 'sentence'
        scores = {}
        for mode in ('whole','unit'):
            metric = _metrics(g['labels'],g[mode],g[mode+'_unknown'])
            metric[denominator+'_auroc'] = metric.pop('auroc')
            metric[denominator+'_average_precision'] = metric.pop('ap')
            scores[mode] = metric
        result[key] = dict(cohort=g['cohort'],original_annotation_granularity=g['granularity'],cases=g['cases'],
            annotation_coverage=dict(g['coverage']),denominator_granularity=denominator,
            response_characters=g['response_characters'],
            released_span_covered_characters=g['released_span_covered_characters'],
            outside_released_span_characters=g['outside_released_span_characters'],
            eval_axis='original_unicode_characters_from_released_token_spans' if denominator == 'character' else 'native_released_sentence_spans',
            historical_bpe_metrics_or_point8_target_comparison_allowed=False,**scores)
    return dict(whole_case_denominator=18,cohorts=result,exploratory_only=True,
        pooled_mixed_granularity_metric_created=False,
        note='Exploratory character metrics are not historical BPE-token metrics or evidence toward the six-task 0.8 target. Token-span complements are negative only with released complete coverage; sentence labels retain native units.')


def evaluate(inputs,observations,natural_panel,controls_panel,output,protocol):
    """Rebuild every retained request and score before releasing native truth."""
    inputs,root,natural,controls,output,protocol = (Path(p).absolute() for p in
        (inputs,observations,natural_panel,controls_panel,output,protocol))
    paths = (inputs,root,natural,controls,output,protocol)
    _require(all(p.resolve() == p for p in paths) and inputs.name == 'inputs.jsonl'
        and protocol.name.casefold() not in shared.ANNOTATION_NAMES,'Unaliased independent evaluation paths required')
    trees = (root,natural,controls,output)
    _require(all(not a.is_relative_to(b) and not b.is_relative_to(a) for i,a in enumerate(trees) for b in trees[i+1:])
        and all(not protocol.is_relative_to(p) for p in trees),'Disjoint evaluation trees required')
    if output.exists():raise FileExistsError('Retained evaluation output must not be overwritten')
    inventory = _inventory(root)
    _snapshot(root,inventory)
    frozen = data.read_json(protocol)
    _require(isinstance(frozen,dict) and not set(frozen)-set(DEFAULT_PROTOCOL)-{'notes','hypothesis_id'}
        and all(type(frozen.get(k)) is type(v) and frozen[k] == v for k,v in DEFAULT_PROTOCOL.items()),
        'Prespecified whole-population evaluation protocol required')
    source,protocol_hash = research_identity(),data.file_hash(protocol)
    samples,readouts,cases,observer_plan,complete = _validate(inputs,root,source)
    panels = []
    for panel,subset in ((natural,samples[:18]),(controls,samples[18:])):
        _require({p.name for p in panel.iterdir()} == _PANEL_FILES|{'seal.json'},'Exact original four-file panel required')
        seal,raw = reader._seal_content(panel,_PANEL_FILES)
        _require(data.file_hash(panel/'inputs.jsonl') == seal['inputs.jsonl'],'Panel input hash changed')
        reader._equal([_identity(s) for s in read_samples(panel/'inputs.jsonl')],[_identity(s) for s in subset],
                      'Panel exact original 18/16 roster changed')
        panels.append((panel,seal,data.text_hash(raw.decode('utf-8'))))
    _snapshot(root,inventory)
    output.mkdir(parents=True)
    data.write_json(output/'plan.json',dict(schema='focused-verdict-evaluation-plan-v1',
        state='evaluator_frozen_before_release',inputs_sha256=observer_plan['original_inputs_sha256'],
        observation_root=str(root),observation_seal_sha256=inventory['hash'],
        original_observation_protocol_sha256=observer_plan['protocol_sha256'],
        child_execution_protocol_sha256=data.file_hash(root/'shared_protocol.json'),
        protocol_provenance_note='Original external protocol hash is retained from the sealed observer plan; exact protocol values and child protocol bytes are independently reconstructed. The reserialized root protocol copy is not asserted byte-identical to the external file.',
        panel_seals=[dict(path=str(p),seal_sha256=h,seal=s) for p,s,h in panels],
        original_ids=[s.id for s in samples],protocol_sha256=protocol_hash,
        evaluator_source_identity_sha256=shared._digest(source),evaluation_material_accessed=False))
    data.write_json(output/'protocol.json',frozen)
    data.write_json(output/'source_identity.json',source)
    frozen_files = {p.name:data.file_hash(p) for p in output.iterdir()}
    completed_identity(source)
    _snapshot(root,inventory)
    for panel,seal,digest in panels:
        _require(data.file_hash(panel/'seal.json') == digest,'Panel seal changed before release')
        reader._equal(data.verify_seal(panel,expected=_PANEL_FILES),seal,'Panel changed before release')
    manifest = reader._manifest(natural,samples[:18],panels[0][1])
    natural_labels = reader._released_rows(natural,samples[:18],manifest)
    control_manifest,control_labels = oldreader._controls(controls,samples[18:],panels[1][1])
    packet_root = output/'cases';packet_root.mkdir()
    records,packets,control_y,control_scores,control_unknown = [],[],[],{'whole':[],'unit':[]},{'whole':[],'unit':[]}
    for index,(sample,readout) in enumerate(zip(samples,readouts,strict=True)):
        packet = dict(schema='focused-verdict-evaluation-case-v1',id=sample.id,source_id=sample.source_id,
            task=sample.task,generator=sample.generator,original_inputs=sample.inputs,readouts=readout,
            observations=dict(whole=cases[readout['whole']['flat_id']],
                units=[cases[u['flat_id']] for u in readout['units']]),
            human_semantic_review=dict(state='pending',complete_human_truth_created=False))
        if index < 18:
            released = natural_labels[index]
            packet.update(released_evaluation_record=released,native_evaluation=_natural(sample,readout,released))
            packets.append(packet)
        else:
            released = control_labels[index-18]
            truth = int(released['construction_logical_outcome'] != 'supported')
            scores = dict(whole=readout['whole']['risk'],unit=max((u['risk'] for u in readout['units']),default=0.0))
            unknown = dict(whole=readout['whole']['unknown_fallback'],unit=any(u['unknown_fallback'] for u in readout['units']))
            packet.update(released_evaluation_record=released,construction_binary_truth=truth,
                          construction_risks=scores,unknown_fallback=unknown)
            control_y.append(truth)
            for mode in scores:
                control_scores[mode].append(scores[mode]);control_unknown[mode].append(unknown[mode])
        path = packet_root/(data.text_hash(sample.id)+'.json');data.write_json(path,packet)
        records.append(dict(id=sample.id,file=str(path.relative_to(output)),sha256=data.file_hash(path)))
    control_index = {s.id:i for i,s in enumerate(samples[18:])}
    pairs = []
    for pair in control_manifest['pairs']:
        indices = [control_index[i] for i in pair['ids']]
        signed = {mode:next((control_scores[mode][b]-control_scores[mode][a] for a in indices for b in indices
            if control_y[a] == 0 and control_y[b] == 1),None) for mode in ('whole','unit')}
        pairs.append(dict(pair_id=pair['pair_id'],ids=pair['ids'],positive_minus_supported_risk=signed,descriptive_only=True))
    summary = dict(whole_case_denominator=34,natural=_natural_summary(packets),
        controls=dict(denominator=16,whole=_metrics(control_y,control_scores['whole'],control_unknown['whole']),
            unit=_metrics(control_y,control_scores['unit'],control_unknown['unit']),pairs=pairs,
            truth='construction only: supported=0; conflict/not_determined=1',empty_response_unit_risk=0.0),
        costs=complete['costs'],coverage=complete['coverage'],model_calls=0,tokenizer_calls=0,
        target_label_selection_performed=False,threshold_sweeps=0,method_promoted=False,six_task_performance_claim=False,
        limitations=list(focus.LIMITATIONS))
    data.write_json(output/'summary.json',summary)
    def unchanged():
        _snapshot(root,inventory);completed_identity(source)
        _require(data.file_hash(inputs) == observer_plan['original_inputs_sha256'] and data.file_hash(protocol) == protocol_hash,
                 'Original inputs/evaluator protocol changed')
        for name,digest in frozen_files.items():_require(data.file_hash(output/name) == digest,'Frozen evaluator metadata changed')
        for panel,seal,digest in panels:
            _require(data.file_hash(panel/'seal.json') == digest,'Panel resealed during evaluation')
            reader._equal(data.verify_seal(panel,expected=_PANEL_FILES),seal,'Panel changed during evaluation')
    unchanged()
    result = dict(schema='focused-verdict-evaluation-complete-v1',state='sealed_exploratory_native_unit_evaluation',
        cases=34,packets=34,modes=['whole','unit'],records=records,evaluation_material_accessed=True,
        model_calls=0,tokenizer_calls=0,observation_root_seal_sha256=inventory['hash'])
    data.write_json(output/'complete.json',result)
    shared._seal(output,[str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()])
    try:data.verify_seal(output);unchanged()
    except Exception:
        (output/'complete.json').unlink(missing_ok=True);(output/'seal.json').unlink(missing_ok=True)
        raise
    return result
