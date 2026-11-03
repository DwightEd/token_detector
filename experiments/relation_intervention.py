"""Root-claimed, fixed48-call role/value path assay on one supplied response.

No target dataset, labels, new answer generation, head/layer selection or risk
score is involved. Every completed unit is independently sealed for later reuse;
an existing output is never automatically restarted after a failure.
"""

from contextlib import nullcontext
from pathlib import Path
import time

import numpy as np

from .data import file_hash, read_json, verify_seal, write_json
from .provenance import completed_identity, research_identity
from .relation_cache_qualification import capture_identity
from .research.relation_intervention import (
    AV_POLICY, BRANCHES, LAYER, WORLD_ORDER, EligibilityError, address, apply_branch,
    attention_diagnostic, capture_rows, output_metrics, prepare_quartet, q_patch, scientific_summary,
)
from .research.streamed_observer import ResourceLimits, SafeTensorStore, StreamedObserver
from .streamed_head_qualification import allocator_cleanup, cache_digest, clone_cache, compare_outputs
from .streamed_qualification import _hook_counts, _path, _resources, verify_assets


ADMISSION_SHA256 = 'a173ce32339ea4f59027ac9bc80b87dcf0536e8875a0b9f229a6d84e85c0b5a3'


def _seal(root):
    write_json(root/'seal.json',{str(p.relative_to(root)):file_hash(p)
        for p in sorted(root.rglob('*')) if p.is_file() and p!=root/'seal.json'})


def _same_bits(left,right):
    return left.dtype==right.dtype and left.shape==right.shape and left.tobytes()==right.tobytes()


def head_prerequisite(root,assets,observer_identity):
    seal=verify_seal(root,required=('plan.json','result.json','bulk-output.npz','cached-output.npz','noop-output.npz'))
    plan,result=read_json(root/'plan.json'),read_json(root/'result.json')
    if (result.get('qualified') is not True or result.get('counts')!={'model_calls':4,'head_calls':3}
            or result.get('cached_vs_noop',{}).get('bitwise_equal') is not True
            or result.get('pristine_cache_unchanged') is not True
            or plan.get('assets')!=assets or plan.get('observer_capture_identity')!=observer_identity):
        raise ValueError('Exact native head/cache/no-op prerequisite is not qualified')
    prior=plan['source_runtime']['source_sha256']
    for key,path in [
        ('experiments.research/streamed_observer.py',Path(__file__).parent/'research/streamed_observer.py'),
        ('experiments/streamed_qualification.py',Path(__file__).with_name('streamed_qualification.py')),
        ('experiments/streamed_head_qualification.py',Path(__file__).with_name('streamed_head_qualification.py'))]:
        if prior.get(key)!=file_hash(path):
            raise ValueError('Head-qualified execution dependency changed')
    return dict(seal_sha256=file_hash(root/'seal.json'),members=seal)


def _arrays(path,values):
    if path.exists():raise FileExistsError('An observed numerical artifact must not be overwritten')
    np.savez_compressed(path,**{key:value for key,value in values.items() if isinstance(value,np.ndarray)})


def _save_cache(path,cache):
    if path.exists():raise FileExistsError('An observed native cache must not be overwritten')
    arrays={}
    for index,pair in enumerate(cache.to_legacy_cache()):
        for kind,value in zip(('key','value'),pair,strict=True):
            arrays[f'{kind}_{index}']=value.detach().cpu().numpy()
    np.savez_compressed(path,**arrays)


def _native_trace(path,trace,cache=None,hidden=None,*,layer=LAYER):
    values={name:value.detach().cpu().numpy() for name,value in trace.items()}
    if cache is not None:
        values.update({name:value.detach().cpu().numpy() for name,value in
            zip(('post_rope_key','value'),cache.to_legacy_cache()[layer],strict=True)})
    if hidden is not None:values['final_hidden']=hidden.detach().cpu().numpy()
    _arrays(path,values)


def run_assay(observer,quartet,output,*,observe=None,counts=None,layer=LAYER):
    """Same bounded engine for tiny CPU audits and the separately claimed run."""
    import torch
    started=time.perf_counter()
    output=Path(output)
    counts=counts if counts is not None else {'model_calls':0,'head_calls':0}
    caches,cache_hashes,baselines,baseline_raw,observations={}, {}, {}, {}, {}
    config=observer.native.config
    base_hooks=_hook_counts(observer.native)
    def model(ids,**kwargs):
        counts['model_calls']+=1
        return observer.hidden(ids,**kwargs)
    def save_unit(identifier,metrics,diagnostic,raw,extra=None):
        unit=output/'units'/identifier
        if not (unit/'attention-state.npz').is_file() or not (unit/'output.npz').is_file():
            raise RuntimeError('A complete unit must retain its phase and full-head observations')
        result=dict(metrics,actual_counts=dict(counts),
            phase={'max_AV_error':diagnostic['av_max_error'],
                   'max_AV_error_over_tolerance':diagnostic['av_max_error_over_tolerance']},
            resources=_resources(observer.device),extra=extra or {})
        write_json(unit/'observation.json',result)
        _seal(unit)
        if observe is not None:observe(identifier,result)
    with torch.inference_mode():
        # All donors are real original-model observations, acquired once only.
        for name in WORLD_ORDER:
            world=quartet['worlds'][name]
            directory=output/'prefills'/name
            directory.mkdir(parents=True,exist_ok=False)
            write_json(directory/'attempt.json',dict(world=name,kind='prefill',counts_before=dict(counts)))
            try:
                with capture_rows(observer,[world['condition_position']],layer=layer) as trace:
                    prefix=model(world['token_ids'][:-1],use_cache=True)
            except BaseException:
                if 'trace' in locals():_native_trace(directory/'partial-native-trace.npz',trace,layer=layer)
                raise
            cache=prefix.past_key_values
            del prefix
            _save_cache(directory/'all-layer-KV.npz',cache)
            _native_trace(directory/'native-trace.npz',trace,cache,layer=layer)
            diagnostic=attention_diagnostic(observer,trace,cache,[world['condition_position']],layer=layer)
            cache_hashes[name]=cache_digest(cache)
            caches[name]=cache
            _arrays(directory/'condition-attention.npz',diagnostic)
            write_json(directory/'receipt.json',dict(cache_sha256=cache_hashes[name],
                actual_counts=dict(counts),condition_position=world['condition_position'],
                instrument_max_AV_error=diagnostic['av_max_error'],
                source_alpha_value=world['alpha_value'],
                per_head=address(diagnostic,world,quartet['worlds'][
                    f"S{1-world['role']}{world['value']}"]['alpha_value']),
                resources=_resources(observer.device)))
            _seal(directory)
            del trace,diagnostic

        def query(name,branch=None):
            world=quartet['worlds'][name]
            role_name=f"S{1-world['role']}{world['value']}"
            value_name=f"S{world['role']}{1-world['value']}"
            identifier=name+'_baseline' if branch is None else name+'_'+branch.name+(
                '' if branch.seed is None else '-'+str(branch.seed))
            unit=output/'units'/identifier
            unit.mkdir(parents=True,exist_ok=False)
            write_json(unit/'attempt.json',dict(world=name,branch='baseline' if branch is None else branch.__dict__,
                                               counts_before=dict(counts)))
            cache=clone_cache(caches[name],config,len(world['token_ids'])-1)
            if branch is not None:
                apply_branch(cache,caches[role_name],caches[value_name],world,branch,layer=layer)
            donor_query=None if branch is None else torch.from_numpy(baselines[role_name]['query'])
            patch=(q_patch(observer,branch,donor_query,layer=layer) if branch is not None else nullcontext())
            # Patch registered first; capture sees the query actually used.
            try:
                with patch,capture_rows(observer,[0],layer=layer) as trace:
                    result=model(world['token_ids'][-1:],past_key_values=cache,use_cache=True)
            except BaseException:
                if 'trace' in locals():_native_trace(unit/'partial-native-trace.npz',trace,cache,layer=layer)
                raise
            hidden=result.last_hidden_state[0,-1:]
            del result
            _native_trace(unit/'native-trace.npz',trace,cache,hidden,layer=layer)
            if hidden.dtype!=torch.float32 or hidden.shape!=(1,config.hidden_size):
                raise RuntimeError('Final native prevalue hidden state changed')
            diagnostic=attention_diagnostic(observer,trace,cache,[world['query_position']],layer=layer)
            _arrays(unit/'attention-state.npz',diagnostic)
            cleanup_before=allocator_cleanup(observer)
            counts['head_calls']+=1
            logits=observer.logits(hidden)
            if logits.shape!=(1,config.vocab_size):
                raise RuntimeError('Native complete-vocabulary head axis changed')
            metrics=output_metrics(logits[0],quartet['candidate_ids'])
            metrics['address']=address(diagnostic,world,quartet['worlds'][role_name]['alpha_value'])
            raw={'hidden':hidden[0].detach().cpu().numpy().copy(),
                 'logits':logits[0].detach().cpu().numpy().copy(),
                 'logp':metrics['candidates']['200']['logp']}
            _arrays(unit/'output.npz',raw)
            write_json(unit/'head-endpoint.json',dict(metrics,actual_counts=dict(counts)))
            del hidden,logits,cache,trace
            cleanup_after=allocator_cleanup(observer)
            if cache_digest(caches[name])!=cache_hashes[name]:
                raise RuntimeError('A query branch mutated its pristine recipient cache')
            save_unit(identifier,metrics,diagnostic,raw,
                      dict(allocator_before=cleanup_before,allocator_after=cleanup_after,
                           recipient=name,role_donor=role_name,value_donor=value_name))
            return metrics,diagnostic,raw

        for name in WORLD_ORDER:
            metrics,diagnostic,raw=query(name)
            baselines[name]=diagnostic
            baseline_raw[name]=raw
            observations[name]={'baseline':metrics,'branches':{}}
        for name in WORLD_ORDER:
            for branch in BRANCHES:
                metrics,diagnostic,raw=query(name,branch)
                if branch.name=='noop' and not compare_outputs(baseline_raw[name],raw)['bitwise_equal']:
                    raise RuntimeError('Scientific stimulus identity no-op failed bitwise equality')
                if branch.name in {'v_role','v_value'} and any(not _same_bits(
                        baselines[name][field],diagnostic[field]) for field in ('query','key','attention')):
                    raise RuntimeError('A V-only branch changed native Q/K/attention')
                key=branch.name+('' if branch.seed is None else '-'+str(branch.seed))
                observations[name]['branches'][key]=metrics
                del diagnostic,raw
        if any(cache_digest(caches[name])!=cache_hashes[name] for name in WORLD_ORDER):
            raise RuntimeError('A donor or recipient pristine cache changed')
    if counts!={'model_calls':48,'head_calls':44}:
        raise RuntimeError('The immutable48-model/44-head budget changed')
    if (_hook_counts(observer.native)!=base_hooks or not observer.resource_report()['all_parameters_meta']
            or observer.resource_report()['active_module'] is not None):
        raise RuntimeError('Intervention hook or module residency leaked')
    summary=scientific_summary(quartet,observations)
    summary.update(actual_counts=dict(counts),elapsed_seconds=time.perf_counter()-started,
                   cache_identity_unchanged=True,all_physical_heads_retained=True,
                   resource_report=observer.resource_report())
    return summary


def run(args):
    """Integrated CLI stage; root owns its spec, memory claim and original run.

    Required: capture_root, model, asset_manifest, head_qualification, admission,
    output. Optional: asset_verification, asset_state, device='cuda:0'. Existing
    outputs fail closed; completed per-unit/cache observations are not rerun.
    """
    import torch
    from transformers import AutoTokenizer
    started=time.perf_counter()
    model,output,capture,head_root,admission=(_path(getattr(args,name)) for name in
        ('model','output','capture_root','head_qualification','admission'))
    if output.exists():raise FileExistsError('Existing assay output must be inspected, never restarted automatically')
    if any(output==root or root in output.parents for root in (model,capture,head_root)):
        raise ValueError('Assay output must be disjoint from immutable input trees')
    if file_hash(admission)!=ADMISSION_SHA256:
        raise ValueError('The independently reviewed scientific admission changed')
    assets=verify_assets(args.asset_manifest,model,
        verification=getattr(args,'asset_verification',None),state=getattr(args,'asset_state',None))
    observer_identity=capture_identity(capture,model/'config.json')
    prerequisite=head_prerequisite(head_root,assets,observer_identity)
    identity=research_identity()
    output.mkdir(parents=True,exist_ok=False)
    observer=None
    counts={'model_calls':0,'head_calls':0}
    completed=[]
    device=torch.device(getattr(args,'device','cuda:0'))
    try:
        tokenizer=AutoTokenizer.from_pretrained(model,use_fast=True,local_files_only=True,trust_remote_code=False)
        quartet=prepare_quartet(tokenizer)
        plan=dict(schema='h017-role-value-intervention-plan-v1',source_runtime=identity,
            admission_sha256=ADMISSION_SHA256,assets=assets,head_prerequisite=prerequisite,
            observer_capture_identity=observer_identity,quartet=quartet,
            layer=LAYER,physical_query_heads=32,kv_heads=8,AV_policy=AV_POLICY,
            coordinate_phase='native postpatch rawQ+native cos/sin vs actual postRoPE cacheKV',
            branches=[branch.__dict__ for branch in BRANCHES],before=_resources(device),
            negative_scope='Only tested numeric-K output mediation; no claim about absence of role information',
            labels_read=False,natural_detection_evaluated=False,automatic_retry=False)
        write_json(output/'plan.json',plan)
        store=SafeTensorStore(model,index_sha256=assets['files']['model.safetensors.index.json']['sha256'],
                              expected_files=assets['files'])
        observer=StreamedObserver(store,read_json(model/'config.json'),
            config_sha256=assets['files']['config.json']['sha256'],device=device,limits=ResourceLimits())
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        def retained(identifier,result):
            completed.append(identifier)
            print(f'RELATION_UNIT {identifier} model={counts["model_calls"]} head={counts["head_calls"]}',flush=True)
        result=run_assay(observer,quartet,output,observe=retained,counts=counts)
        result.update(schema='h017-role-value-intervention-result-v1',
            original_generator_internal_state=False,labels_read=False,
            elapsed_seconds=time.perf_counter()-started,after=_resources(device))
        write_json(output/'result.json',result)
    except BaseException as error:
        write_json(output/'failure.json',dict(type=type(error).__name__,message=str(error),
            actual_counts=counts,completed_units=completed,eligibility_failure=isinstance(error,EligibilityError),
            resources=_resources(device if torch.cuda.is_initialized() else torch.device('cpu')),
            automatic_retry=False,elapsed_seconds=time.perf_counter()-started))
        raise
    finally:
        if observer is not None:observer.close()
    completed_identity(identity)
    verify_seal(head_root,expected=prerequisite['members'])
    if file_hash(admission)!=ADMISSION_SHA256 or verify_assets(args.asset_manifest,model,
            verification=getattr(args,'asset_verification',None),state=getattr(args,'asset_state',None))!=assets:
        raise RuntimeError('Frozen assay dependencies changed')
    _seal(output)
    return result
