"""Freeze matched whole/unit focus observations without reading annotations."""

from dataclasses import asdict
import json
from pathlib import Path
import re

from . import data, verdict_qualification as shared
from .datasets import read_samples
from .provenance import completed_identity, research_identity
from .research import focused_verdict as focus


DEFAULT_PROTOCOL = dict(schema='focused-verdict-qualification-v1',
    contract_version=focus.CONTRACT_VERSION,purpose='observations_only',
    context_policy=shared.DEFAULT_PROTOCOL['context_policy'],expected_original_cases=34,
    original_inputs_sha256=None,modes=['whole','unit'],partition_policy='response_units_v1',
    unit_selection='all_exact_character_partition_units',max_input_tokens=4096,
    labels=list(focus.LABELS),permutations=6,use_cache=False,logits_to_keep=1,
    attention_implementation='sdpa',risk_definition='1-minus-uniform-six-supported',
    unknown_risk=1.0,token_projection_performed=False)


def _protocol(path):
    value=data.read_json(path)
    if (not isinstance(value,dict)
            or set(value)-set(DEFAULT_PROTOCOL)-{'notes','hypothesis_id'}
            or any(type(value.get(k)) is not type(v) or value[k]!=v
                   for k,v in DEFAULT_PROTOCOL.items() if k!='original_inputs_sha256')
            or not isinstance(value.get('original_inputs_sha256'),str)
            or re.fullmatch('[0-9a-f]{64}',value['original_inputs_sha256']) is None):
        raise ValueError('Prespecified whole/unit focused observation protocol required')
    return value


def _partition(response):
    units=focus.response_units(response)
    cursor=0
    for unit in units:
        if (unit['start']!=cursor or not cursor<unit['end']<=len(response)
                or unit['text']!=response[cursor:unit['end']]):
            raise ValueError('Focus units must preserve every original character exactly')
        cursor=unit['end']
    if cursor!=len(response) or ''.join(unit['text'] for unit in units)!=response:
        raise ValueError('Focus partition omitted original response characters')
    return units


def _prepare(samples):
    originals,flat_rows,flat_records=[],[],[]
    for sample in samples:
        original_inputs=sample.inputs
        context,transformation=shared._context(original_inputs)
        units=_partition(sample.response)
        original=dict(id=sample.id,source_id=sample.source_id,task=sample.task,
            generator=sample.generator,input_text_sha256={k:data.text_hash(v) for k,v in original_inputs.items()},
            pre_generation_context_sha256=data.text_hash(context),context_transformation=transformation,
            response_characters=len(sample.response),units=[])
        for mode,index,span,text in [('whole',None,[0,len(sample.response)],sample.response)]+[
                ('unit',index,[unit['start'],unit['end']],unit['text']) for index,unit in enumerate(units)]:
            flat_id='focus-'+shared._digest(dict(original_id=sample.id,mode=mode,unit_index=index,focus_span=span))
            flat_rows.append(dict(asdict(sample),id=flat_id))
            record=dict(flat_id=flat_id,original_id=sample.id,source_id=sample.source_id,mode=mode,
                unit_index=index,focus_span=span,focus_text=text,
                input_text_sha256=dict(original['input_text_sha256']))
            flat_records.append(record)
            if mode=='whole':original['whole_focus_id']=flat_id
            else:original['units'].append(dict(flat_id=flat_id,index=index,start=span[0],end=span[1],text=text))
        originals.append(original)
    if len({row['id'] for row in flat_rows})!=len(flat_rows):
        raise ValueError('Focus metadata identities collided')
    return originals,flat_rows,flat_records


def _risk(case,unknown_risk):
    unscored=case['verdict'].get('status')=='unknown'
    scores=case['verdict'].get('class_scores')
    return dict(risk=unknown_risk if unscored else 1.0-scores['supported'],
        unknown_fallback=unscored,class_scores=scores,
        status='unknown' if unscored else 'scored')


def qualify(inputs,protocol,output,backend,model_identity):
    """Collect all 34 matched whole responses and every exact-text unit.

    The whole-focus condition uses the same focus prompt as individual units.
    These new matched requests do not import H003 whole-answer observations.
    """
    inputs,protocol,output=(Path(p).absolute() for p in (inputs,protocol,output))
    if (any(p.resolve()!=p for p in (inputs,protocol,output))
            or inputs.name!='inputs.jsonl' or protocol.name.casefold() in shared.ANNOTATION_NAMES
            or 'outputs' not in output.parts or inputs.is_relative_to(output)
            or protocol.is_relative_to(output)):
        raise ValueError('Unaliased original inputs and independent protocol/retained output required')
    if output.exists():raise FileExistsError('Existing focused run requires inspection; automatic retry is forbidden')
    frozen=_protocol(protocol)
    input_hash,protocol_hash=data.file_hash(inputs),data.file_hash(protocol)
    if input_hash!=frozen['original_inputs_sha256']:
        raise ValueError('Original whole-population input hash differs from frozen protocol')
    samples=list(read_samples(inputs))
    if len(samples)!=frozen['expected_original_cases']:
        raise ValueError('The whole original 34-case population is required')
    identity=shared._model_identity(model_identity)
    if getattr(backend,'model_identity',None)!=identity:
        raise ValueError('Backend and frozen model identities differ')
    source=research_identity()
    originals,flat_rows,flat_records=_prepare(samples)
    child_protocol=dict(frozen,expected_cases=len(flat_rows),max_forward_passes=6*len(flat_rows))
    output.mkdir(parents=True)
    prepared=output/'prepared_flat';prepared.mkdir()
    flat_inputs=prepared/'inputs.jsonl'
    with flat_inputs.open('x',encoding='utf-8',newline='\n') as stream:
        for row in flat_rows:
            stream.write(json.dumps(row,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
    child_path=output/'shared_protocol.json'
    data.write_json(child_path,child_protocol)
    flat_hash=data.file_hash(flat_inputs)
    plan=dict(schema='focused-verdict-qualification-plan-v1',state='frozen_before_observation',
        original_inputs_path=str(inputs),original_inputs_sha256=input_hash,protocol_sha256=protocol_hash,
        protocol=frozen,source_identity=source,model_identity=identity,
        prepared_inputs_sha256=flat_hash,prepared_cases=len(flat_rows),logical_requests=6*len(flat_rows),
        max_forward_passes=6*len(flat_rows),records=originals,flat_records=flat_records,
        evaluation_material_accessed=False,metadata_ids_are_model_inputs=False,
        matched_whole_is_new_necessary_granularity_control=True,h003_observations_imported=False,
        limitations=list(focus.LIMITATIONS))
    data.write_json(output/'plan.json',plan)
    data.write_json(output/'protocol.json',frozen)
    data.write_json(output/'source_identity.json',source)
    data.write_json(output/'model_identity.json',identity)
    shared._seal(prepared,['inputs.jsonl'])
    frozen_files={p.name:data.file_hash(p) for p in output.iterdir() if p.is_file()}
    metadata={row['flat_id']:row for row in flat_records}
    def builder(sample,assignment):
        record=metadata[sample.id]
        if {k:data.text_hash(v) for k,v in sample.inputs.items()}!=record['input_text_sha256']:
            raise ValueError('Prepared focus input changed from its original full strings')
        return focus.build_request(sample.source,sample.response,shared._context(sample.inputs)[0],
                                   record['focus_span'],assignment)
    def loader(path):
        value=data.read_json(path)
        if value!=child_protocol:raise ValueError('Frozen all-unit shared protocol changed')
        return value
    if (data.file_hash(inputs)!=input_hash or data.file_hash(protocol)!=protocol_hash
            or data.file_hash(flat_inputs)!=flat_hash):
        raise ValueError('Frozen focused inputs/protocol changed before observation')
    completed=shared.qualify(flat_inputs,child_path,output/'observations',backend,identity,
        sample_request_builder=builder,protocol_loader=loader)
    completed_identity(source)
    if (data.file_hash(inputs)!=input_hash or data.file_hash(protocol)!=protocol_hash
            or data.file_hash(flat_inputs)!=flat_hash
            or any(data.file_hash(output/name)!=digest for name,digest in frozen_files.items())):
        raise ValueError('Frozen focused population/metadata changed during observation')
    data.verify_seal(prepared,expected={'inputs.jsonl'})
    data.verify_seal(output/'observations')
    scored={}
    for record in completed['records']:
        case=data.read_json(output/'observations'/record['file'])
        scored[case['id']]=_risk(case,frozen['unknown_risk'])
    readouts=[]
    for original in originals:
        readouts.append(dict(id=original['id'],source_id=original['source_id'],
            whole=dict(flat_id=original['whole_focus_id'],**scored[original['whole_focus_id']]),
            units=[dict(unit,**scored[unit['flat_id']]) for unit in original['units']]))
    unknown_units=sum(unit['unknown_fallback'] for row in readouts for unit in row['units'])
    complete=dict(schema='focused-verdict-qualification-complete-v1',state='observations_sealed_not_evaluated',
        coverage=dict(original_cases=34,whole_focuses=34,unit_focuses=len(flat_rows)-34,
            flat_cases=len(flat_rows),scored_focuses=completed['coverage']['scored_cases'],
            unknown_focuses=completed['coverage']['unknown_cases'],unknown_units=unknown_units,
            empty_responses=sum(not sample.response for sample in samples)),
        costs=completed['costs'],records=readouts,conservative_unknown_risk=frozen['unknown_risk'],
        source_identity_sha256=shared._digest(source),model_identity_sha256=shared._digest(identity),
        evaluation_material_accessed=False,token_projection_performed=False,auroc_computed=False,
        h003_observations_imported=False)
    data.write_json(output/'complete.json',complete)
    shared._seal(output,[str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()])
    data.verify_seal(output)
    return complete
