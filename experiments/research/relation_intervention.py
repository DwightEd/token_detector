"""A frozen role/value quartet and native-path interventions, never risk scores.

Only the declared synthetic revenue schema licenses ownership. Natural prose,
global matching and target-label head selection are outside this module.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import json
from types import SimpleNamespace

import numpy as np

from myproject.inputs import encode_response, encode_source
from .relation_inventory import align_source_spans, source_inventory


LAYER = 15
WORLD_ORDER = ('S00', 'S10', 'S01', 'S11')
RESPONSE = 'Alpha revenue was 200.'
INSTRUCTION = 'Summarize the revenue records faithfully.\nSource:\n'
ATTENTION_DEADBAND = 1e-5
AV_POLICY = {'absolute': .002, 'relative': .0001}


class EligibilityError(ValueError):
    pass


@dataclass(frozen=True)
class Branch:
    name: str
    seed: int | None = None


BRANCHES = (Branch('q_role'), Branch('k_role'), Branch('v_role'), Branch('qk_role'),
            Branch('v_value'), Branch('noop'), Branch('off_target_random_k', 0),
            *(Branch('random_k', seed) for seed in (0, 1, 2)))


def _single_token(offsets, start, stop):
    indices = [i for i, (left, right) in enumerate(offsets) if left < stop and right > start]
    if len(indices) != 1 or list(offsets[indices[0]]) != [start, stop]:
        raise EligibilityError('A numeric occurrence must be one complete original token')
    return indices[0]


def prepare_quartet(tokenizer, *, maximum=256):
    """Tokenizer-only eligibility: do not rescue names, format or segmentation."""
    if not getattr(tokenizer, 'is_fast', False):
        raise EligibilityError('The frozen quartet requires exact Fast tokenizer offsets')
    answer = encode_response(tokenizer, RESPONSE)
    other_text = RESPONSE.replace('200', '100')
    other = encode_response(tokenizer, other_text)
    start = RESPONSE.index('200')
    target_index = _single_token(answer['offsets'], start, start+3)
    other_index = _single_token(other['offsets'], start, start+3)
    if (target_index != other_index or target_index < 1
            or answer['answer_ids'][:target_index] != other['answer_ids'][:other_index]):
        raise EligibilityError('Candidate100/200 do not share the complete prediction prefix')
    candidate_ids = {'100': other['answer_ids'][other_index], '200': answer['answer_ids'][target_index]}
    if len(set(candidate_ids.values())) != 2:
        raise EligibilityError('Candidate numeral IDs must differ')
    subject_rows = [i for i, (left, right) in enumerate(answer['offsets']) if left < 5 and right > 0]
    if not subject_rows:
        raise EligibilityError('The response condition has no original token occurrence')
    prefix = answer['answer_ids'][:target_index]
    special = set(tokenizer.all_special_ids) | {
        key for key, value in tokenizer.added_tokens_decoder.items() if value.special}
    worlds = {}
    for name in WORLD_ORDER:
        role, value = int(name[1]), int(name[2])
        companies = ('Beta', 'Alpha') if role else ('Alpha', 'Beta')
        numbers = (200, 100) if value else (100, 200)
        facts = dict(zip(companies, numbers, strict=True))
        source = json.dumps({'revenue_by_company': facts}, separators=(', ', ': '))
        prompt = INSTRUCTION + source
        pack = encode_source(tokenizer, prompt, source)
        chat = tokenizer.apply_chat_template([dict(role='user', content=prompt)],
                    tokenize=False, add_generation_prompt=True)
        rendered_prompt = prompt if prompt in chat else prompt.strip()
        if chat.count(rendered_prompt) != 1 or rendered_prompt.count(source) != 1:
            raise EligibilityError('The rendered source/prompt occurrence is ambiguous')
        shift = chat.index(rendered_prompt)
        source_start = shift + rendered_prompt.index(source)
        encoded = tokenizer(chat, add_special_tokens=False, return_offsets_mapping=True)
        if encoded['input_ids'] != pack['prompt_with_source']:
            raise EligibilityError('Supplemental offsets differ from the mainline prompt axis')
        inventory = source_inventory(source, name)
        fields = inventory['fields']
        if {tuple(f['path']) for f in fields} != {
                ('revenue_by_company', 'Alpha'), ('revenue_by_company', 'Beta')}:
            raise EligibilityError('Source is outside the declared two-company schema')
        alignment = align_source_spans(inventory, source, chat, source_start,
            encoded['offset_mapping'], encoded['input_ids'], pack['prompt_with_source'], pack['source_mask'])
        mapped = {m['occurrence_id']: m for m in alignment['occurrences'] if m['kind']=='fields'}
        occurrences = []
        for field in fields:
            match = mapped[field['occurrence_id']]
            if (match['status'] != 'contained' or len(match['contained_tokens']) != 1
                    or match['boundary_crossing_tokens']):
                raise EligibilityError('Source numeral has ambiguous original BPE ownership')
            position = match['contained_tokens'][0]
            if list(encoded['offset_mapping'][position]) != [source_start+x for x in field['span']]:
                raise EligibilityError('Source numeral token includes unlicensed characters')
            if encoded['input_ids'][position]!=candidate_ids[field['value']]:
                raise EligibilityError('Source/response numeral token identity differs')
            occurrences.append(dict(company=field['path'][-1], value=int(field['value']),
                                    position=position, path=field['path'], source_span=field['span']))
        occurrences.sort(key=lambda row: row['position'])
        full_ids = pack['prompt_with_source'] + prefix
        condition = len(pack['prompt_with_source']) + subject_rows[-1]
        query = len(full_ids)-1
        off_target = [i for i, ((left, right), token) in enumerate(zip(encoded['offset_mapping'],
                      encoded['input_ids'], strict=True)) if shift <= left < right <= source_start
                      and token not in special][:2]
        if len(full_ids)>maximum or condition>=query or len(off_target)!=2:
            raise EligibilityError('Frozen input length/condition/off-target positions are ineligible')
        worlds[name] = dict(id=name, role=role, value=value, source=source, prompt=prompt,
            token_ids=full_ids, source_mask=pack['source_mask']+[False]*len(prefix),
            prompt_tokens=len(pack['prompt_with_source']), query_position=query,
            condition_position=condition, occurrences=occurrences, off_target_positions=off_target,
            alpha_value=facts['Alpha'], relation_license='declared_constructed_revenue_schema_only',
            response_prefix_ids=prefix, alignment=alignment)
    reference = worlds['S00']
    for name, world in worlds.items():
        if (world['query_position']!=reference['query_position']
                or world['condition_position']!=reference['condition_position']
                or world['off_target_positions']!=reference['off_target_positions']
                or [r['position'] for r in world['occurrences']] != [r['position'] for r in reference['occurrences']]):
            raise EligibilityError('Four worlds do not preserve the frozen absolute token positions')
        donor = worlds[f"S{1-world['role']}{world['value']}"]
        if any(world['token_ids'][r['position']] != donor['token_ids'][r['position']]
               for r in world['occurrences']):
            raise EligibilityError('A role-only donor changed a literal numeral ID')
    return dict(schema='h017-role-value-quartet-v1', response=RESPONSE, candidate_ids=candidate_ids,
                target_answer_index=target_index, answer_ids=answer['answer_ids'], worlds=worlds,
                budget={'model_calls':48, 'head_calls':44}, scientific_population='one_constructed_quartet')


def matched_noise(delta, seed):
    """Match each KV-head/position norm; seed/zero handling are prespecified."""
    import torch
    generator = torch.Generator(device='cpu').manual_seed(seed)
    noise = torch.randn(delta.shape, generator=generator, dtype=torch.float32).to(delta.device)
    target = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
    scale = torch.linalg.vector_norm(noise, dim=-1, keepdim=True)
    return noise * torch.where(scale>0,target/scale,torch.zeros_like(target))


def apply_branch(cache, role_cache, value_cache, world, branch, *, layer=LAYER):
    """Mutate only an independently cloned recipient cache at the fixed path."""
    import torch
    key, value = cache.to_legacy_cache()[layer]
    role_key, role_value = role_cache.to_legacy_cache()[layer]
    value_payload = value_cache.to_legacy_cache()[layer][1]
    positions = torch.tensor([r['position'] for r in world['occurrences']], device=key.device)
    with torch.inference_mode():
        if branch.name in {'k_role','qk_role'}:
            key.index_copy_(2, positions, role_key.index_select(2, positions))
        elif branch.name=='v_role':
            value.index_copy_(2, positions, role_value.index_select(2, positions))
        elif branch.name=='v_value':
            value.index_copy_(2, positions, value_payload.index_select(2, positions))
        elif branch.name=='noop':
            for tensor in (key,value):
                tensor.index_copy_(2, positions, tensor.index_select(2, positions).clone())
        elif branch.name in {'random_k','off_target_random_k'}:
            delta = role_key.index_select(2, positions)-key.index_select(2, positions)
            noise = matched_noise(delta, branch.seed)
            target = (torch.tensor(world['off_target_positions'], device=key.device)
                      if branch.name=='off_target_random_k' else positions)
            key.index_copy_(2,target,key.index_select(2,target)+noise)
        elif branch.name!='q_role':
            raise ValueError('Unregistered intervention branch')


@contextmanager
def q_patch(observer, branch, donor_query, *, layer=LAYER):
    handle = None
    if branch.name in {'q_role','qk_role','noop'}:
        def replace(module,args,output):
            import torch
            if output.dtype!=torch.float32:
                raise RuntimeError('Native query precision changed')
            if branch.name=='noop':return output.clone()
            if donor_query.shape!=output.shape or donor_query.dtype!=torch.float32:
                raise RuntimeError('Role donor query shape/precision differs')
            return donor_query.to(output.device).clone()
        handle = observer.layers[layer].self_attn.q_proj.register_forward_hook(replace)
    try:
        yield
    finally:
        if handle is not None:handle.remove()


@contextmanager
def capture_rows(observer, rows, *, layer=LAYER):
    """Install after Q patch: capture actual Q, native cos/sin and pre-O headAV."""
    import torch
    record, handles = {}, []
    attention = observer.layers[layer].self_attn
    def selected(value):
        if value.dtype!=torch.float32:
            raise RuntimeError('Native diagnostic value is not FP32 before any conversion')
        return value[:,rows].detach().cpu().clone()
    def positions(module,args,kwargs):
        cosine,sine=kwargs['position_embeddings']
        record['cosine'],record['sine']=selected(cosine),selected(sine)
    handles.append(attention.register_forward_pre_hook(positions,with_kwargs=True))
    handles.append(attention.q_proj.register_forward_hook(lambda m,a,o:record.update(query=selected(o))))
    handles.append(attention.o_proj.register_forward_pre_hook(lambda m,a:record.update(head=selected(a[0]))))
    try:
        yield record
    finally:
        for handle in handles:handle.remove()


def attention_diagnostic(observer, record, cache, query_positions, *, layer=LAYER,
                         variant='faithful', absolute=.002, relative=.0001):
    """Official native post-RoPE phase; bad variants are synthetic test controls."""
    import torch
    implementation=observer.implementation
    module=observer.layers[layer].self_attn
    heads,kv,width=observer.head_layout(layer)
    query=record['query'].reshape(1,len(query_positions),heads,width).transpose(1,2)
    cosine,sine=record['cosine'],record['sine']
    if variant!='omit_rope':
        query=implementation.apply_rotary_pos_emb(query,query,cosine,sine)[0]
    key,value=(v.detach().cpu() for v in cache.to_legacy_cache()[layer])
    positions=torch.tensor(query_positions)[:,None]
    allowed=torch.arange(key.shape[2])[None]<=positions
    mask=torch.zeros((1,1,len(query_positions),key.shape[2]),dtype=torch.float32)
    if variant!='omit_causal':mask.masked_fill_(~allowed[None,None],-torch.inf)
    if variant=='interleave_gqa':
        mapping=torch.arange(heads)%kv
        key,value=key[:,mapping],value[:,mapping]
        module=SimpleNamespace(num_key_value_groups=1,training=False,scaling=module.scaling)
    elif variant not in {'faithful','omit_rope','omit_causal'}:
        raise ValueError('Unknown diagnostic test variant')
    with torch.inference_mode():
        av,attention=implementation.eager_attention_forward(module,query,key,value,mask,
                                                scaling=module.scaling,dropout=0.)
    actual=av.reshape_as(record['head'])
    error=(actual-record['head']).abs()
    tolerance=absolute+relative*record['head'].abs()
    if not torch.isfinite(actual).all() or (error>tolerance).any():
        raise RuntimeError('Post-RoPE attention instrument does not reproduce native headAV')
    return dict(query=record['query'].numpy(),rotated_query=query.numpy(),key=key.numpy(),
                value=value.numpy(),attention=attention[0].numpy(),head=record['head'].numpy(),
                query_positions=np.asarray(query_positions,dtype=np.int64),
                av_max_error=float(error.max()),av_max_error_over_tolerance=float((error/tolerance).max()))


def output_metrics(logits, candidate_ids):
    import torch
    if logits.dtype!=torch.float32 or logits.ndim!=1 or not torch.isfinite(logits).all():
        raise ValueError('Full-vocabulary endpoint must be finite FP32')
    logp=logits.log_softmax(-1)
    result={}
    for name,index in candidate_ids.items():
        result[name]=dict(token_id=index,logit=float(logits[index]),logp=float(logp[index]),
                          probability=float(logp[index].exp()),rank=int((logits>logits[index]).sum())+1)
    return dict(candidates=result,margin=result['200']['logit']-result['100']['logit'],
                candidate_mass=sum(x['probability'] for x in result.values()),top1=int(logits.argmax()))


def address(diagnostic, world, target_value):
    positions=[r['position'] for r in world['occurrences']]
    values=[r['value'] for r in world['occurrences']]
    weights=diagnostic['attention'][:,0,positions]
    target=values.index(target_value)
    contrast=weights[:,target]-weights[:,1-target]
    source=np.asarray(world['source_mask'][:diagnostic['attention'].shape[-1]],dtype=bool)
    return dict(per_head_numeric_attention=weights.tolist(),per_head_address_contrast=contrast.tolist(),
                address=float(contrast.mean()),per_head_numeric_mass=weights.sum(-1).tolist(),
                per_head_source_mass=diagnostic['attention'][:,0,source].sum(-1).tolist(),
                basis='unconditional_all_physical_heads_target_occurrence_minus_other',
                target_value=target_value,target_position=positions[target],
                other_value=values[1-target],other_position=positions[1-target])


def decision_deadband(baseline,changed):
    return .008+.0001*sum(abs(row['candidates'][v]['logit']) for row in (baseline,changed) for v in ('100','200'))


def scientific_summary(quartet, observations):
    """Frozen single-quartet comparisons, not a detector or population estimate."""
    rows=[]
    for name,world in quartet['worlds'].items():
        baseline=observations[name]['baseline']
        branches=observations[name]['branches']
        donor=quartet['worlds'][f"S{1-world['role']}{world['value']}"]
        direction=1 if donor['alpha_value']==200 else -1
        effect=direction*(branches['k_role']['margin']-baseline['margin'])
        noise=max(abs(branches[f'random_k-{seed}']['margin']-baseline['margin']) for seed in (0,1,2))
        off=abs(branches['off_target_random_k-0']['margin']-baseline['margin'])
        address_shift=branches['k_role']['address']['address']-baseline['address']['address']
        value_donor=quartet['worlds'][f"S{world['role']}{1-world['value']}"]
        value_direction=1 if value_donor['alpha_value']==200 else -1
        value_effect=value_direction*(branches['v_value']['margin']-baseline['margin'])
        rows.append(dict(id=name,role_effect=effect,noise_max=noise,off_target=off,
            address_shift=address_shift,role_pass=effect>max(decision_deadband(baseline,branches['k_role']),noise,off)
            and address_shift>ATTENTION_DEADBAND,
            value_effect=value_effect,value_pass=value_effect>decision_deadband(baseline,branches['v_value'])))
    baselines={name:row['baseline']['margin'] for name,row in observations.items()}
    reorder=max(abs(baselines['S00']-baselines['S11']),abs(baselines['S10']-baselines['S01']))
    legible=all(row['baseline']['top1'] in quartet['candidate_ids'].values() for row in observations.values())
    return dict(rows=rows,reorder_difference=reorder,reorder_pass=reorder<min(r['role_effect'] for r in rows),
                task_legible=legible,scoped_mechanism_gate=legible and all(r['role_pass'] and r['value_pass'] for r in rows)
                and reorder<min(r['role_effect'] for r in rows),
                negative_scope='Rejects tested block15 numeric-K final-output mediation only; not absence of encoded role information',
                natural_detection_evaluated=False,auroc=None,population='one_constructed_quartet')
