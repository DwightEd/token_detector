"""Freeze source-only role questions and literal completions before observations.

This is a narrow, previously exposed constructed-panel qualification of the
untested source-reconstruction core. It is not a natural-data detector.
"""
from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path
import re

from . import data, answer_completion as completion, verdict_evaluation as reader
from .datasets import read_samples
from .provenance import completed_identity, research_identity
from .research import source_reconstruction as kernel
from .semantic_qualification import ANNOTATION_NAMES, _digest, _model_identity, _seal

DEFAULT_PROTOCOL=dict(schema='source-reconstruction-qualification-v1',
    purpose='source_only_completion_observations',expected_original_cases=48,
    expected_unique_sources=8,expected_unique_queries=16,expected_completions=32,
    expected_response_slots=96,original_inputs_sha256=None,max_input_tokens=4096,
    source_grammar='h007_explicit_two_person_two_event_v1',
    query_policy='source_only_no_response_values',candidate_order='sorted_complete_source_names',
    completion_contract_version=completion.CONTRACT_VERSION,
    eos_policy='complete_literal_candidate_then_one_tokenizer_eos',
    use_cache=False,output_tokens=0,attention_implementation='sdpa',
    length_normalization=False,temperature=1.0,token_projection_performed=False,
    population_previously_exposed=True)

_META=('completion_contract_version','raw_candidate','whitespace_policy','prefix_ids',
    'prefix_tokens','candidate_token_ids','candidate_tokens','eos_token_id','input_ids',
    'input_tokens','completion_eos_appended_to_input','target_token_ids','target_tokens',
    'target_prediction_positions','logits_to_keep','prefix_alignment_verified',
    'candidate_roundtrip_verified','causal_shift_verified')


def _require(condition,message):
    if not condition:raise ValueError(message)


def _protocol(path):
    value=data.read_json(path)
    _require(isinstance(value,dict) and not set(value)-set(DEFAULT_PROTOCOL)-{'notes','hypothesis_id'}
        and all(type(value.get(k)) is type(v) and value[k]==v
            for k,v in DEFAULT_PROTOCOL.items() if k!='original_inputs_sha256')
        and isinstance(value.get('original_inputs_sha256'),str)
        and re.fullmatch('[0-9a-f]{64}',value['original_inputs_sha256']) is not None,
        'Prespecified complete source-reconstruction population required')
    return value


def _prepare(samples):
    _require(len(samples)==48 and len({s.id for s in samples})==48,'All48 original cases required')
    originals,sources,queries=[],{},{}
    for sample in samples:
        source_key=data.text_hash(sample.source)
        if source_key not in sources:
            prepared=kernel.source_queries(sample.source)
            _require(prepared['status']=='ok','Source grammar not admitted; core not yet tested')
            sources[source_key]=deepcopy(prepared)
            for query in prepared['queries']:
                _require(query['source']==sample.source and query['candidates']==sorted(set(query['candidates']))
                    and len(query['candidates'])==2,'Source-only complete canonical candidate domain required')
                key=query['query_key']
                _require(key not in queries or queries[key]==query,'Conflicting source query identity')
                queries[key]=deepcopy(query)
        alignment=kernel.align_response(sample.source,sample.response)
        _require(alignment['status']=='ok' and len(alignment['slots'])==2,
            'Response frontend not admitted; no model observations may be interpreted')
        for slot in alignment['slots']:
            _require(slot['query_key'] in queries and sample.response[slot['start']:slot['end']]==slot['text'],
                'Exact original response slot was not preserved')
        originals.append(dict(sample=asdict(sample),source_key=source_key,alignment=alignment,
            input_text_sha256={k:data.text_hash(v) for k,v in sample.inputs.items()}))
    _require(len(sources)==8 and len(queries)==16
        and sum(len(o['alignment']['slots']) for o in originals)==96,
        'All8 sources/16 questions/96 slots required without selection')
    candidates=[dict(query_key=key,candidate=candidate,messages=deepcopy(queries[key]['messages']))
        for key in sorted(queries) for candidate in queries[key]['candidates']]
    return dict(originals=originals,sources=sources,queries=queries,candidates=candidates)


def _metadata(value,candidate):
    _require(isinstance(value,dict) and set(value)==set(_META),'Complete literal-token framing required')
    _require(value['completion_contract_version']==completion.CONTRACT_VERSION
        and value['raw_candidate']==candidate and candidate and candidate==candidate.strip()
        and value['whitespace_policy']==completion.WHITESPACE_POLICY
        and all(value[k] is True for k in ('prefix_alignment_verified','candidate_roundtrip_verified','causal_shift_verified'))
        and value['completion_eos_appended_to_input'] is False,'Candidate literal/prefix/EOS contract changed')
    for key in ('prefix_ids','candidate_token_ids','input_ids','target_token_ids','target_prediction_positions'):
        _require(isinstance(value[key],list) and value[key]
            and all(type(x) is int and x>=0 for x in value[key]),'Invalid token metadata: '+key)
    p,k=value['prefix_ids'],value['candidate_token_ids']
    eos=value['eos_token_id']
    _require(type(eos) is int and eos>=0 and eos not in k
        and value['input_ids']==p+k and value['target_token_ids']==k+[eos]
        and value['target_prediction_positions']==list(range(len(p)-1,len(p)+len(k))),
        'Full first-token-through-EOS causal alignment changed')
    expected=dict(prefix_tokens=len(p),candidate_tokens=len(k),input_tokens=len(p)+len(k),
        target_tokens=len(k)+1,logits_to_keep=len(k)+1)
    _require(all(type(value.get(key)) is int and value[key]==count for key,count in expected.items()),
        'Candidate/EOS token counts changed')
    return deepcopy(value)


def _execution(row,metadata,protocol_hash,identity,source):
    return dict(messages=deepcopy(row['messages']),candidate=row['candidate'],model_identity=identity,
        protocol_sha256=protocol_hash,source_identity_sha256=_digest(source),preflight=deepcopy(metadata),
        scoring=dict(use_cache=False,output_tokens=0,complete_candidate_and_eos=True,
            full_vocab_normalizer_dtype='float64',length_normalization=False,temperature=1.0))


def _plan(inputs,input_hash,protocol_hash,frozen,identity,source,prepared,executions):
    return dict(schema='source-reconstruction-plan-v1',state='frozen_before_model_observation',
        original_inputs_path=str(inputs),original_inputs_sha256=input_hash,protocol_sha256=protocol_hash,
        protocol=frozen,model_identity=identity,source_identity=source,prepared=prepared,
        executions=executions,evaluation_material_accessed=False,response_used_in_model_requests=False,
        natural_data_coverage_established=False)


def _response(raw,metadata):
    _require(isinstance(raw,dict) and 'status' not in raw
        and all(raw.get(key)==value and type(raw.get(key)) is type(value) for key,value in metadata.items()),
        'Completion preflight changed after freezing')
    n=metadata['target_tokens']
    vectors=[raw.get(key) for key in ('target_token_logits','full_vocab_logsumexp','target_token_log_probabilities')]
    _require(all(isinstance(v,list) and len(v)==n and
        all(type(x) in (int,float) and math.isfinite(x) for x in v) for v in vectors),
        'Every first-name-token through EOS observation must be finite and retained')
    logits,norms,logp=vectors
    _require(all(abs(a-b-c)<=1e-9 and c<=1e-10 for a,b,c in zip(logits,norms,logp,strict=True))
        and type(raw.get('joint_log_probability')) in (int,float)
        and math.isfinite(raw['joint_log_probability'])
        and abs(math.fsum(logp)-raw['joint_log_probability'])<=1e-9,
        'Full-vocabulary causal token probabilities/joint sum changed')
    vocab=raw.get('vocab_size')
    _require(type(vocab) is int and vocab>max(metadata['input_ids']+metadata['target_token_ids'])
        and raw.get('logits_shape')==[1,n,vocab] and raw.get('full_vocab_normalizer_dtype')=='float64'
        and type(raw.get('actual_forward_passes')) is int and raw['actual_forward_passes']==1
        and type(raw.get('output_tokens')) is int and raw['output_tokens']==0
        and type(raw.get('seconds')) in (int,float) and math.isfinite(raw['seconds']) and raw['seconds']>=0,
        'Completion vocabulary/shape/forward costs changed')
    admission=raw.get('resource_admission')
    _require(isinstance(admission,dict) and admission.get('device')=='cuda:0'
        and all(type(admission.get(k)) is int for k in
            ('available_ram_bytes','free_cuda_bytes','cuda_allocator_limit_bytes'))
        and admission['available_ram_bytes']>=2*1024**3 and admission['free_cuda_bytes']>=3*1024**3
        and admission['cuda_allocator_limit_bytes']==3*1024**3,
        'Frozen 3B resource admission is missing or invalid')
    allocated,reserved=raw.get('peak_cuda_allocated_bytes'),raw.get('peak_cuda_reserved_bytes')
    _require(type(allocated) is int and type(reserved) is int
        and 0<=allocated<=reserved<=3*1024**3,'Invalid allocator peak bytes')
    return dict(deepcopy(raw),status='scored')


def _inventory(root,keys=None,observed=None,*,has_freeze=True,has_complete=False,has_seal=False,pending=None):
    """Validate every allowed member/alias before any payload hash."""
    root=Path(root)
    _require(root.resolve()==root and root.is_dir(),'Unaliased reconstruction tree required')
    names={'plan.json','protocol.json','model_identity.json','source_identity.json','requests'}
    if has_freeze:names.add('freeze.json')
    if has_complete:names.add('complete.json')
    if has_seal:names.add('seal.json')
    _require({p.name for p in root.iterdir()}==names,'Unexpected reconstruction root members')
    files=set()
    for name in names-{'requests'}:
        reader._member(root,name)
        files.add(name)
    requests=reader._member(root,'requests',directory=True)
    actual={p.name for p in requests.iterdir()}
    _require(len(actual)==32 and all(re.fullmatch('[0-9a-f]{64}',k) for k in actual)
        and (keys is None or actual==set(keys)),'Complete exact32 request inventory required')
    observed=actual if observed is None else set(observed)
    _require(observed<=actual and (pending is None or pending in actual-observed),'Invalid cache progress inventory')
    for key in actual:
        child=reader._member(requests,key,directory=True)
        expected=({'request.json','backend_raw.json','response.json','seal.json'} if key in observed else
            {'request.json','backend_raw.json','response.json'} if key==pending else {'request.json'})
        _require({p.name for p in child.iterdir()}==expected,'Unexpected completion child members')
        for name in expected:
            files.add(str(reader._member(child,name).relative_to(root)))
    return files


def _completion(prepared,executions,responses,frozen):
    lookup={(e['query_key'],e['candidate']):e['key'] for e in executions}
    predictions={}
    for key,query in prepared['queries'].items():
        values={candidate:responses[lookup[key,candidate]]['joint_log_probability']
            for candidate in query['candidates']}
        distribution=kernel.candidate_distribution(query,values)
        _require(distribution['status']=='ok' and math.isfinite(distribution['log_candidate_normalizer'])
            and distribution['log_candidate_normalizer']<=1e-9,
            'Complete candidate-plus-EOS events must form a valid finite subdistribution')
        predictions[key]=dict(query_key=key,distribution=distribution,
            completion_keys={candidate:lookup[key,candidate] for candidate in query['candidates']},
            candidate_token_counts={candidate:responses[lookup[key,candidate]]['candidate_tokens']
                for candidate in query['candidates']})
    records=[]
    for original in prepared['originals']:
        slots=[]
        for slot in original['alignment']['slots']:
            prediction=predictions[slot['query_key']]['distribution']
            probability=prediction['probabilities'].get(slot['text'])
            _require(probability is not None,'Original response value is outside the admitted source domain')
            slots.append(dict(slot,risk=1.0-probability,unknown_fallback=False,
                candidate_conditional_probability=probability,calibrated_truth_probability=False))
        records.append(dict(id=original['sample']['id'],source_id=original['sample']['source_id'],
            source_key=original['source_key'],slots=slots,
            case_max_slot_risk=max(s['risk'] for s in slots),unknown_fallback=False))
    return dict(schema='source-reconstruction-complete-v1',state='observations_sealed_not_evaluated',
        queries=predictions,records=records,
        coverage=dict(original_cases=48,unique_sources=8,unique_queries=16,unique_completions=32,
            response_slots=96,unknown_cases=0,unknown_slots=0),
        costs=dict(actual_forward_passes=sum(r['actual_forward_passes'] for r in responses.values()),
            input_tokens=sum(r['input_tokens'] for r in responses.values()),
            teacher_forced_target_tokens=sum(r['target_tokens'] for r in responses.values()),
            output_tokens=0,seconds_forward=math.fsum(r['seconds'] for r in responses.values())),
        response_values_exposed_to_qa=False,evaluation_material_accessed=False,
        token_projection_performed=False,natural_task_performance_claim=False,method_promoted=False)


def qualify(inputs,protocol,output,backend,model_identity):
    inputs,protocol,output=(Path(p).absolute() for p in (inputs,protocol,output))
    _require(all(p.resolve()==p for p in (inputs,protocol,output))
        and inputs.name=='inputs.jsonl'
        and protocol.name.casefold() not in ANNOTATION_NAMES|{'construction_truth.jsonl'}
        and not protocol.is_relative_to(inputs.parent) and 'outputs' in output.parts
        and not inputs.is_relative_to(output) and not output.is_relative_to(inputs.parent)
        and not protocol.is_relative_to(output),'Independent unaliased inference/protocol/output paths required')
    if output.exists():raise FileExistsError('Retained source-QA run requires inspection, not automatic retry')
    frozen,identity=_protocol(protocol),_model_identity(model_identity)
    input_hash,protocol_hash=data.file_hash(inputs),data.file_hash(protocol)
    _require(input_hash==frozen['original_inputs_sha256'],'Complete original input hash changed')
    _require(getattr(backend,'model_identity',None)==identity,'Backend model identity differs')
    source=research_identity();prepared=_prepare(list(read_samples(inputs)))
    executions={}
    for row in prepared['candidates']:
        metadata=_metadata(deepcopy(backend.validate_completion(deepcopy(row['messages']),row['candidate'])),row['candidate'])
        _require(metadata['input_tokens']<=frozen['max_input_tokens'],'Complete population exceeds the untruncated token budget')
        execution=_execution(row,metadata,protocol_hash,identity,source);key=_digest(execution)
        _require(key not in executions,'Unexpected duplicate semantic completion')
        executions[key]=dict(query_key=row['query_key'],candidate=row['candidate'],key=key,execution=execution)
    _require(len(executions)==32,'All32 unique completions required before model calls')
    prefixes={}
    for row in executions.values():
        prefix=row['execution']['preflight']['prefix_ids']
        _require(row['query_key'] not in prefixes or prefixes[row['query_key']]==prefix,
            'Candidates of one source question changed the conditional prefix')
        prefixes[row['query_key']]=prefix
    _require(len({e['execution']['preflight']['eos_token_id'] for e in executions.values()})==1,
        'All complete candidate events require one fixed tokenizer EOS')
    plan=_plan(inputs,input_hash,protocol_hash,frozen,identity,source,prepared,list(executions.values()))
    output.mkdir(parents=True)
    for name,value in [('plan.json',plan),('protocol.json',frozen),('model_identity.json',identity),('source_identity.json',source)]:
        data.write_json(output/name,value)
    for key,row in executions.items():
        root=output/'requests'/key;root.mkdir(parents=True);data.write_json(root/'request.json',row['execution'])
    paths=_inventory(output,executions,set(),has_freeze=False)
    immutable={name:data.file_hash(output/name) for name in paths}
    data.write_json(output/'freeze.json',immutable);freeze_hash=data.file_hash(output/'freeze.json')
    child_snapshots={};completion_digest=None;root_sealed=False
    def unchanged():
        _inventory(output,executions,child_snapshots,has_complete=completion_digest is not None,has_seal=root_sealed)
        _require(data.file_hash(inputs)==input_hash and data.file_hash(protocol)==protocol_hash
            and data.file_hash(output/'freeze.json')==freeze_hash
            and getattr(backend,'model_identity',None)==identity
            and all(data.file_hash(output/name)==digest for name,digest in immutable.items()),
            'Frozen source-only request population/model/protocol changed')
        _require(all(data.file_hash(output/name)==digest
            for snapshot in child_snapshots.values() for name,digest in snapshot.items()),
            'Previously sealed raw completion/response cache changed')
        if completion_digest is not None:
            _require(data.file_hash(output/'complete.json')==completion_digest,'Final completion changed')
        completed_identity(source)
    responses={}
    for key,row in executions.items():
        unchanged();execution=row['execution'];messages=deepcopy(execution['messages'])
        raw=backend.score_completion(messages,row['candidate'])
        root=output/'requests'/key;data.write_json(root/'backend_raw.json',{'value':raw})
        _require(messages==execution['messages'],'Backend mutated a frozen source-QA request')
        response=_response(raw,execution['preflight']);responses[key]=response
        data.write_json(root/'response.json',response)
        _inventory(output,executions,child_snapshots,pending=key)
        _seal(root,['request.json','backend_raw.json','response.json'])
        _inventory(output,executions,set(child_snapshots)|{key})
        data.verify_seal(root,expected={'request.json','backend_raw.json','response.json'})
        _require(data.read_json(root/'request.json')==execution
            and data.read_json(root/'backend_raw.json')=={'value':raw}
            and data.read_json(root/'response.json')==response,'Disk cache does not match validated observations')
        child_snapshots[key]={str((root/name).relative_to(output)):data.file_hash(root/name)
            for name in ('request.json','backend_raw.json','response.json','seal.json')}
    unchanged()
    if hasattr(backend,'verify_frozen_identity'):backend.verify_frozen_identity()
    result=_completion(prepared,list(executions.values()),responses,frozen)
    data.write_json(output/'complete.json',result)
    paths=_inventory(output,executions,child_snapshots,has_complete=True)
    completion_digest=data.file_hash(output/'complete.json')
    try:
        unchanged();_seal(output,sorted(paths));root_sealed=True
        _inventory(output,executions,child_snapshots,has_complete=True,has_seal=True)
        data.verify_seal(output,expected=paths);unchanged()
    except BaseException:
        (output/'complete.json').unlink(missing_ok=True);(output/'seal.json').unlink(missing_ok=True)
        raise
    return result
