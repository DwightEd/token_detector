"""CPU-only source-frozen evaluation of the designed response-burden controls."""
from collections import defaultdict
import json
from pathlib import Path
from . import data, designed_focus_qualification as focused
from . import verdict_evaluation as reader, verdict_qualification as shared
from . import focused_verdict_evaluation as prior
from .datasets import Sample, read_samples, read_jsonl
from .provenance import completed_identity, research_identity
from .research import focused_verdict as focus

DEFAULT_PROTOCOL=dict(schema='designed-focus-evaluation-v1',purpose='construction_control_diagnosis',
    required_cases=48,modes=['whole','target_first_unit'],burdens=[0,4,12],
    model_calls_allowed=False,tokenizer_calls_allowed=False,target_label_selection_allowed=False,
    primary_effect='(target_D12-target_D0)-(whole_D12-whole_D0)',
    risk='one-minus-uniform-six-supported',unknown_risk=1.0,natural_task_auroc_claim=False)
_HELPERS=('experiments/designed_focus_qualification.py','experiments/focused_verdict_qualification.py',
    'experiments.research/focused_verdict.py','experiments/verdict_qualification.py',
    'experiments.research/semantic_verdict.py','experiments/semantic_qualification.py',
    'experiments/data.py','experiments/datasets.py')
_require=prior._require
_inventory=prior._inventory
_snapshot=prior._snapshot

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
    _require(len(samples) == 48 and len({s.id for s in samples}) == 48, 'Whole original 48-case population required')
    originals,flat_rows,flat_records = focused._prepare(samples)
    flat_bytes = ''.join(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
                         for row in flat_rows).encode('utf-8')
    flat_inputs = root/'prepared_flat/inputs.jsonl'
    _require(flat_inputs.read_bytes() == flat_bytes, 'Prepared full input strings/order changed')
    flat_hash = data.file_hash(flat_inputs)
    child_protocol = dict(protocol,expected_cases=len(flat_rows),max_forward_passes=6*len(flat_rows))
    reader._equal(data.read_json(root/'shared_protocol.json'),child_protocol,'Shared protocol changed')
    expected = dict(schema='designed-focus-qualification-plan-v1',state='frozen_before_observation',
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
    complete = dict(schema='designed-focus-qualification-complete-v1',state='observations_sealed_not_evaluated',
        coverage=dict(original_cases=48,whole_focuses=48,unit_focuses=len(flat_rows)-48,flat_cases=len(flat_rows),
            scored_focuses=completion['coverage']['scored_cases'],unknown_focuses=completion['coverage']['unknown_cases'],
            unknown_units=sum(u['unknown_fallback'] for r in readouts for u in r['units']),
            empty_responses=sum(not s.response for s in samples)),costs=completion['costs'],records=readouts,
        conservative_unknown_risk=1.0,source_identity_sha256=shared._digest(source),
        model_identity_sha256=shared._digest(model),evaluation_material_accessed=False,
        token_projection_performed=False,auroc_computed=False,h003_observations_imported=False)
    reader._equal(data.read_json(root/'complete.json'),complete,'Parent risk/cost/coverage completion changed')
    return samples,readouts,cases,plan,complete


def _design(path, samples):
    value=data.read_json(path)
    rows=value.get('records') if isinstance(value,dict) else None
    _require(isinstance(value,dict) and value.get('schema')=='h007-burden-controls-plan-v1' and isinstance(rows,list)
        and len(rows)==48 and [r.get('id') for r in rows]==[s.id for s in samples],
        'Complete ordered construction design required')
    cells=set()
    for sample,row in zip(samples,rows,strict=True):
        cell=row['factor_cell']
        key=tuple(cell[name] for name in ('relation_family','vocabulary','source_direction','response_direction','distractor_count'))
        _require(key not in cells,'Duplicate construction factor cell');cells.add(key)
        unit=focus.response_units(sample.response)[0]
        _require(row['target_span']==[unit['start'],unit['end']] and row['target_text']==unit['text']
            and row['source_id']==sample.source_id and row['whole_span']==[0,len(sample.response)]
            and row['input_text_sha256']=={k:data.text_hash(v) for k,v in sample.inputs.items()},
            'Designed first unit or original strings changed')
    expected={(f,v,s,r,b) for f in ('winner','sole_signer') for v in ('lex1','lex2')
              for s in (0,1) for r in (0,1) for b in (0,4,12)}
    _require(cells==expected,'All prescribed factor cells required')
    by_scenario=defaultdict(list)
    for sample,row in zip(samples,rows,strict=True):
        c=row['factor_cell'];by_scenario[(c['relation_family'],c['vocabulary'],c['source_direction'],c['response_direction'])].append((sample,row))
    for group in by_scenario.values():
        _require(len({s.source for s,r in group})==1 and len({r['target_text'] for s,r in group})==1,
                 'Burden changed source facts or target text')
    return rows


def _analysis(design, readouts, truth):
    _require(len(truth)==48 and [r.get('id') for r in truth]==[r['id'] for r in design],
             'Complete ordered construction truth required')
    cells=defaultdict(dict);packets=[]
    for row,score,label in zip(design,readouts,truth,strict=True):
        _require(row['id']==score['id']==label['id'] and label.get('source_id')==row['source_id']
            and label.get('target_outcome') in ('supported','contradicted')
            and label.get('whole_outcome')==label['target_outcome']
            and label.get('source_sha256')==row['input_text_sha256']['source']
            and label.get('response_sha256')==row['input_text_sha256']['response']
            and label.get('target_span')==row['target_span'], 'Construction truth join differs')
        cell=row['factor_cell'];positive=int(label['target_outcome']=='contradicted')
        _require(label.get('filler_outcomes')==['supported']*cell['distractor_count']
            and label.get('target_benchmark_labels_used') is False
            and label.get('construction_rule')=='entity_event_bijections_equal_iff_source_direction_equals_response_direction',
            'Construction filler or source-only truth contract changed')
        _require(positive==int(cell['source_direction']!=cell['response_direction']),
                 'Construction truth contradicts frozen bijection rule')
        risk=dict(whole=score['whole'],target_first_unit=score['units'][0])
        key=tuple(cell[n] for n in ('relation_family','vocabulary','source_direction','distractor_count'))
        _require(positive not in cells[key],'Duplicate paired construction outcome')
        cells[key][positive]=risk
        packets.append(dict(id=row['id'],factor_cell=cell,label=positive,risks=risk))
    _require(len(cells)==24 and all(set(pair)=={0,1} for pair in cells.values()),
             'All24 paired cells required')
    pairs=[]
    for key,pair in cells.items():
        pairs.append(dict(relation_family=key[0],vocabulary=key[1],source_direction=key[2],distractor_count=key[3],
            differences={mode:pair[1][mode]['risk']-pair[0][mode]['risk'] for mode in DEFAULT_PROTOCOL['modes']}))
    results={}
    for mode in DEFAULT_PROTOCOL['modes']:
        results[mode]={}
        for burden in (0,4,12):
            selected=[p for p in packets if p['factor_cell']['distractor_count']==burden]
            contrasts=[p['differences'][mode] for p in pairs if p['distractor_count']==burden]
            results[mode][str(burden)]=dict(prior._metrics([p['label'] for p in selected],
                [p['risks'][mode]['risk'] for p in selected],
                [p['risks'][mode]['unknown_fallback'] for p in selected]),
                paired_cells=len(contrasts),mean_wrong_minus_supported_risk=sum(contrasts)/len(contrasts),
                correctly_ordered_pairs=sum(d>0 for d in contrasts),tied_pairs=sum(d==0 for d in contrasts))
    means={mode:{b:results[mode][b]['mean_wrong_minus_supported_risk'] for b in ('0','4','12')}
           for mode in DEFAULT_PROTOCOL['modes']}
    interaction=(means['target_first_unit']['12']-means['target_first_unit']['0'])-(means['whole']['12']-means['whole']['0'])
    families={family:[p for p in pairs if p['relation_family']==family] for family in ('winner','sole_signer')}
    return dict(cases=48,paired_cells=24,by_mode_and_burden=results,pairs=pairs,by_family_pairs=families,
        primary_interaction=interaction,primary_effect=DEFAULT_PROTOCOL['primary_effect'],
        threshold_search_performed=False,target_benchmark_labels_accessed=False,method_promoted=False,
        natural_task_performance_claim=False,limitations=['Constructed explicit relations only; no original-generator causality or benchmark efficacy.',
            'Four lexical scenarios with correlated factor cells; no large-sample significance or model selection.']),packets


def evaluate(inputs, design, construction, observations, output, protocol):
    """Rebuild every frozen request/readout before opening construction truth."""
    inputs,design,construction,root,output,protocol=(Path(p).absolute() for p in
        (inputs,design,construction,observations,output,protocol))
    _require(all(p.resolve()==p for p in (inputs,design,construction,root,output,protocol))
        and inputs.name=='inputs.jsonl' and design.name=='plan.json'
        and construction.name=='construction_truth.jsonl', 'Unaliased original construction paths required')
    trees=(root,inputs.parent,construction.parent,output)
    _require('outputs' in output.parts and design==inputs.parent/'plan.json'
        and all(not a.is_relative_to(b) and not b.is_relative_to(a)
            for index,a in enumerate(trees) for b in trees[index+1:])
        and all(not protocol.is_relative_to(p) for p in trees)
        and protocol.name.casefold() not in shared.ANNOTATION_NAMES|{'construction_truth.jsonl'},
        'Independent observation/inference/evaluation/label trees and external protocol required')
    if output.exists():raise FileExistsError('Retained evaluation must not be overwritten')
    inventory=_inventory(root);_snapshot(root,inventory)
    frozen=data.read_json(protocol)
    _require(isinstance(frozen,dict) and not set(frozen)-set(DEFAULT_PROTOCOL)-{'notes','hypothesis_id'}
        and all(type(frozen.get(k)) is type(v) and frozen[k]==v for k,v in DEFAULT_PROTOCOL.items()),
        'Prespecified construction evaluation protocol required')
    source=research_identity()
    samples,readouts,cases,observer_plan,completed=_validate(inputs,root,source)
    designed=_design(design,samples)
    hashes={str(p):data.file_hash(p) for p in (inputs,design,protocol)}
    output.mkdir(parents=True)
    data.write_json(output/'plan.json',dict(schema='designed-focus-evaluation-plan-v1',
        state='frozen_before_construction_truth',source_identity=source,protocol=frozen,
        source_hashes=hashes,observation_initial_seal_sha256=inventory['hash'],
        construction_truth_path=str(construction),construction_truth_accessed=False,model_calls_allowed=False))
    data.write_json(output/'freeze.json',{'plan.json':data.file_hash(output/'plan.json')})
    freeze_hash=data.file_hash(output/'freeze.json')
    # Only now may construction labels, or their hashes, be opened.
    label_parent=construction.parent
    _require(label_parent.resolve()==label_parent and label_parent.is_dir()
        and {p.name for p in label_parent.iterdir()}=={'construction_truth.jsonl','seal.json'},
        'Independent complete construction truth directory required')
    reader._member(label_parent,'construction_truth.jsonl');reader._member(label_parent,'seal.json')
    label_seal,label_bytes=reader._seal_content(label_parent,{'construction_truth.jsonl'})
    reader._verify_observation_snapshot(label_parent,{'construction_truth.jsonl'},label_seal,data.text_hash(label_bytes.decode('utf-8')),[])
    truth=list(read_jsonl(construction))
    summary,packets=_analysis(designed,readouts,truth)
    summary.update(costs=completed['costs'],coverage=completed['coverage'],model_calls=0)
    data.write_json(output/'summary.json',summary);data.write_json(output/'cases.json',packets)
    def unchanged():
        _snapshot(root,inventory);completed_identity(source)
        _require(data.file_hash(output/'freeze.json')==freeze_hash and
            data.file_hash(output/'plan.json')==data.read_json(output/'freeze.json')['plan.json']
            and all(data.file_hash(Path(p))==digest for p,digest in hashes.items()),
            'Frozen CPU plan or original design changed')
        reader._verify_observation_snapshot(label_parent,{'construction_truth.jsonl'},label_seal,data.text_hash(label_bytes.decode('utf-8')),[])
    unchanged()
    data.write_json(output/'complete.json',dict(schema='designed-focus-evaluation-complete-v1',
        state='sealed_construction_burden_diagnosis',cases=48,model_calls=0,
        evaluation_only_after_predictions_frozen=True,natural_task_performance_claim=False))
    try:
        shared._seal(output,[str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()])
        data.verify_seal(output)
        unchanged()
    except BaseException:
        for name in ('complete.json','seal.json'):
            (output/name).unlink(missing_ok=True)
        raise
    return summary

