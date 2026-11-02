"""H004 crossed source rendering / rule intervention using the shared collector."""
from copy import deepcopy
from pathlib import Path

from . import data, verdict_qualification as collector
from .datasets import read_samples
from .observation_reuse import effective_model_identity, equivalence_key, load_donor
from .provenance import research_identity, completed_identity
from .research import semantic_verdict as verdict, typed_source
from .semantic_qualification import _context, _digest, _seal, _artifact_path
from .semantic_qualification import ANNOTATION_NAMES

CONDITIONS = {'A': (False, False), 'B': (True, False),
              'C': (False, True), 'D': (True, True)}
NULL_RULE = (
    ' Interpret structured field types explicitly: a null or None value means '
    'the field is not reported, and cannot support either a positive or a '
    'negative assertion. A boolean false or False is an explicitly negative '
    'value for that field, not missing evidence. Preserve the binding of every '
    'value to its own field, entity and time. Do not transfer a value or missing '
    'status from a different field. These rules do not add any source facts.'
)
BASE_PROTOCOL = dict(schema='typed-verdict-factorial-v1', purpose='observations_only',
    conditions=['A','B','C','D'], expected_cases=34, max_input_tokens=4096,
    permutations=6, max_new_forward_passes=612, labels=['A','B','C'],
    source_transform_id=typed_source.SOURCE_TRANSFORM_ID, null_rule=NULL_RULE,
    use_cache=False, logits_to_keep=1, attention_implementation='sdpa')
BASE_PROTOCOL['donor_inputs_sha256']=[
    '1bf9a26ad06ac350988302999776e7288f9da570d35316fd6b27ca21e36e8baf',
    '7bf1a14dff728aa367ce66664413cdb1d58bea9500218dec4173144a412f6a19']


def _protocol(path):
    value=data.read_json(path)
    if (not isinstance(value,dict) or set(value)-set(BASE_PROTOCOL)-{'notes','hypothesis_id'}
            or any(type(value.get(k)) is not type(v) or value.get(k)!=v for k,v in BASE_PROTOCOL.items())):
        raise ValueError('H004 must preserve the complete prespecified 2x2 protocol')
    return value


def build_request(source,response,context,assignment,condition):
    if condition not in CONDITIONS:raise ValueError('Unknown H004 condition')
    typed,rule=CONDITIONS[condition]
    prepared=typed_source.prepare_source(source)
    request=verdict.build_request(prepared['source_text'] if typed else source,response,context,assignment)
    if rule:request['messages'][0]['content']+=NULL_RULE
    return request


class ReusingScorer:
    """Import exact old observations; mark new calls and within-run reuse separately."""
    def __init__(self,backend,identity,donors):
        if getattr(backend,'model_identity',None)!=identity:
            raise ValueError('Backend and declared model identities differ')
        self.backend,self.model_identity,self.donors=backend,identity,donors
        self.observed={}

    def _execution(self,messages,labels):
        return dict(messages=messages,labels=list(labels),model_identity=self.model_identity,
             scoring=dict(use_cache=False,logits_to_keep=1,full_prefill=True,output_tokens=0),
             admission=dict(max_input_tokens=4096))

    def _old(self,messages,labels):
        key=equivalence_key(self._execution(messages,labels))
        return key,self.donors.get(key)

    def count_input_tokens(self,messages):
        _,old=self._old(messages,verdict.LABELS)
        return old['response']['input_tokens'] if old else self.backend.count_input_tokens(messages)

    def validate_labels(self,messages,labels):
        _,old=self._old(messages,labels)
        if old:
            return {k:deepcopy(old['response'][k]) for k in
              ('input_tokens','label_token_ids','label_decodes','prefix_alignment_verified')}
        return self.backend.validate_labels(messages,labels)

    def score_labels(self,messages,labels):
        key,old=self._old(messages,labels)
        if old:
            raw={k:deepcopy(v) for k,v in old['response'].items() if k!='status'}
            raw['observation_origin']=dict(mode='donor_reuse',equivalence_key=key,
                                           donor=old['provenance'])
            return raw
        if key in self.observed:
            raw=deepcopy(self.observed[key])
            raw['observation_origin']=dict(mode='in_run_reuse',equivalence_key=key)
            return raw
        raw=self.backend.score_labels(messages,labels)
        if 'status' in raw or 'observation_origin' in raw:
            raise ValueError('Backend must return raw instrument fields only')
        self.observed[key]=deepcopy(raw)
        return dict(raw,observation_origin=dict(mode='observed',equivalence_key=key))

    def verify_frozen_identity(self):
        if hasattr(self.backend,'verify_frozen_identity'):self.backend.verify_frozen_identity()


def _costs(output):
    origins={};new={};borrowed=set();within=0;rejections=0
    for condition in CONDITIONS:
        for path in (output/condition/'requests').glob('*/response.json'):
            row=data.read_json(path)
            if row['status']=='input_rejected':rejections+=1;continue
            origin=row['observation_origin'];key=origin['equivalence_key'];mode=origin['mode']
            origins.setdefault(key,mode)
            if mode=='observed':
                if key in new:raise ValueError('An equivalent request was forwarded twice')
                new[key]=row
            elif mode=='donor_reuse':borrowed.add(key)
            elif mode=='in_run_reuse':within+=1
            else:raise ValueError('Unknown observation origin')
    return dict(logical_requests=34*4*6,unique_equivalent_observations=len(origins),
       new_forward_passes=len(new),borrowed_unique_observations=len(borrowed),
       within_run_reused_artifacts=within,input_rejected_artifacts=rejections,
       new_forward_seconds=sum(r['seconds'] for r in new.values()),
       new_input_tokens=sum(r['input_tokens'] for r in new.values()),output_tokens=0,
       imported_forward_cost_is_not_new_compute=True)


def qualify(inputs,protocol,output,backend,model_identity,donor_roots):
    """Freeze all conditions and origins before new observation; no annotation I/O."""
    inputs,protocol,output=map(lambda p:Path(p).absolute(),(inputs,protocol,output))
    if (inputs.resolve()!=inputs or protocol.resolve()!=protocol or output.resolve()!=output
            or protocol.name.casefold() in ANNOTATION_NAMES
            or inputs.name!='inputs.jsonl' or 'outputs' not in output.parts):
        raise ValueError('Original inputs.jsonl and retained outputs are required')
    if output.exists():raise FileExistsError('H004 output exists: inspect/reuse, never restart blindly')
    frozen=_protocol(protocol);source=research_identity()
    samples=list(read_samples(inputs))
    if len(samples)!=34:raise ValueError('All18 natural and16 constructed original cases required')
    if len(donor_roots)!=2:raise ValueError('Both fixed H003 donor cohorts are required')
    input_hash,protocol_hash=data.file_hash(inputs),data.file_hash(protocol)
    # Donor cohorts are bound by their own archived input digests and root seals.
    donors={};donor_bindings=[];original_samples=[]
    for donor_root,expected_hash in zip(donor_roots,frozen['donor_inputs_sha256'],strict=True):
        root=Path(donor_root).absolute()
        index=load_donor(root,model_identity,expected_hash)
        # load_donor inventories paths and seals before this metadata is opened.
        plan=data.read_json(root/'plan.json')
        original_samples.extend(read_samples(Path(plan['inputs_path'])))
        for key,item in index.items():
            if key in donors and donors[key]['response']!=item['response']:
                raise ValueError('Equivalent donor requests have conflicting observations')
            donors.setdefault(key,item)
        donor_bindings.append(dict(path=str(root),seal_sha256=data.file_hash(root/'seal.json'),
                                   inputs_sha256=plan['inputs_sha256']))
    def original_identity(sample):
        return dict(id=sample.id,source_id=sample.source_id,task=sample.task,
                    generator=sample.generator,inputs=sample.inputs)
    if (len({s.id for s in samples})!=34 or
            [original_identity(s) for s in samples]!=[original_identity(s) for s in original_samples]):
        raise ValueError('H004 requires the exact ordered18+16 original population, not equivalent text substitutes')
    roster=[];all_executions={}
    for sample in samples:
        context,transformation=_context(sample.inputs)
        prepared=typed_source.prepare_source(sample.source)
        row=dict(id=sample.id,source_id=sample.source_id,input_text_sha256={k:data.text_hash(v) for k,v in sample.inputs.items()},
                 context_transformation=transformation,source_transformation=prepared,conditions={})
        for condition in CONDITIONS:
            requests=[]
            for assignment in verdict.label_assignments():
                request=build_request(sample.source,sample.response,context,assignment,condition)
                execution=dict(messages=request['messages'],labels=list(verdict.LABELS),model_identity=model_identity,
                  scoring=dict(use_cache=False,logits_to_keep=1,full_prefill=True,output_tokens=0),
                  admission=dict(max_input_tokens=4096))
                key=equivalence_key(execution);all_executions.setdefault(key,execution)
                requests.append(dict(equivalence_key=key,label_assignment=assignment,
                    origin='H003_donor' if key in donors else 'new_or_exact_within_run_reuse'))
                if condition=='A' and key not in donors:
                    raise ValueError('Original A must reuse exact H003 observations; no A rerun allowed')
            row['conditions'][condition]=requests
        roster.append(row)
    new_keys=set(all_executions)-set(donors)
    if len(new_keys)>frozen['max_new_forward_passes']:raise ValueError('New-forward budget exceeded')
    output.mkdir(parents=True)
    plan=dict(schema='typed-verdict-factorial-plan-v1',protocol=frozen,inputs_sha256=input_hash,
      protocol_sha256=protocol_hash,source_identity=source,model_identity=model_identity,
      effective_model_identity=effective_model_identity(model_identity),donors=donor_bindings,records=roster,
      expected_new_unique_requests=len(new_keys),evaluation_material_accessed=False)
    data.write_json(output/'plan.json',plan)
    scorer=ReusingScorer(backend,model_identity,donors)
    # Preflight every novel request before the first model forward of any arm.
    for execution in all_executions.values():collector._preflight(scorer,execution)
    for condition in CONDITIONS:
        child_protocol=dict(frozen,condition=condition,expected_cases=34,max_forward_passes=204)
        path=output/(condition+'-protocol.json');data.write_json(path,child_protocol)
        def loader(p,expected=child_protocol):
            value=data.read_json(p)
            if value!=expected:raise ValueError('Frozen child protocol changed')
            return value
        def builder(s,r,c,a,arm=condition):return build_request(s,r,c,a,arm)
        collector.qualify(inputs,path,output/condition,scorer,model_identity,
                           request_builder=builder,protocol_loader=loader)
    completed_identity(source)
    if data.file_hash(inputs)!=input_hash or data.file_hash(protocol)!=protocol_hash:
        raise ValueError('Frozen H004 inputs/protocol changed')
    for binding in donor_bindings:
        if data.file_hash(Path(binding['path'])/'seal.json')!=binding['seal_sha256']:
            raise ValueError('Donor archive resealed during H004')
        data.verify_seal(Path(binding['path']))
    costs=_costs(output)
    if costs['new_forward_passes']>len(new_keys):raise ValueError('Repeated model forwards')
    complete=dict(schema='typed-verdict-factorial-complete-v1',state='observations_sealed_not_evaluated',
      cases=34,conditions=list(CONDITIONS),costs=costs,source_identity_sha256=_digest(source),
      evaluation_material_accessed=False,auroc_computed=False)
    data.write_json(output/'complete.json',complete)
    _seal(output,[str(p.relative_to(output)) for p in output.rglob('*') if p.is_file()])
    return complete
