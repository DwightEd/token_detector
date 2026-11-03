"""Five frozen source-scope worlds, sequential native observations and finite evaluation.

This is a mechanism diagnostic on one existing response, not a detector. It uses
the qualified original-weight streamed observer, native instrumentation and mainline
text encoders. Old BF16 scalar probabilities are not a same-run numerical oracle.
"""

from contextlib import ExitStack
from pathlib import Path
import ast
import json
import shutil
import time

import numpy as np

from .data import file_hash, read_json, text_hash, verify_seal, write_json
from .datasets import Sample
from .natural_relation_observation import query_axis, token_endpoints
from .provenance import completed_identity, research_identity
from .relation_cache_qualification import capture_identity
from .relation_intervention import _arrays, _seal, head_prerequisite
from .research.relation_intervention import AV_POLICY, attention_diagnostic, capture_rows
from .research.relation_scope import WORLD_ORDER, prepare_worlds, scientific_summary
from .research.streamed_observer import ResourceLimits, SafeTensorStore, StreamedObserver
from .streamed_head_qualification import allocator_cleanup
from .streamed_qualification import _hook_counts, _resources, verify_assets


ADMISSION_SHA256 = 'a815dd39f2bca07b7bd4f54d281174b40dfaff4a972865194aba634a49738c43'
LICENSE_SHA256 = '71180505cefbf8c5197c9371f417f7c88f71e37de978f5d29df8550ef1ee664d'
RESOURCE_SHA256 = 'a63d86b6871bd59ad3a081867a6a1949bd1a44b76b7c5c1c849178ada80bd072'
AUDIT_SHA256 = '6672f7398168fe218e6dbd260d3edc1cf1d94f5ec906571d87040c0a471d9cde'
AMENDMENT_SHA256 = 'c3b4d3530a317614a7c79aee8072317bea6b5c47e9c1f9e5647c5a94ef0f9f1d'


def _profiles(diagnostic, world):
    """All physical-head masses on overlapping exact source anchors, no scores."""
    attention = diagnostic['attention']
    result = {name: attention[..., row['overlap_tokens']].sum(-1)
              for name, row in world['source_anchors'].items()}
    result['source'] = attention[..., np.asarray(world['source_mask'], dtype=bool)].sum(-1)
    return result


def observe_world(observer, world, candidates, output, *, counts, target_index=55):
    """One native call, layerwise bounded host copies, then one complete head.

    Shared by the admitted run and tiny CPU instrumentation/failure checks. A
    partial attempt is retained and cannot be restarted in the same directory.
    """
    import torch
    output = Path(output)
    output.mkdir(exist_ok=False)
    raw = output/'raw'
    raw.mkdir()
    traces, native = {}, None
    hooks = _hook_counts(observer.native)
    rows = world['capture_positions']
    n, p, t = len(world['token_ids']), world['prompt_tokens'], len(world['answer_ids'])
    if (world['token_ids'][p:] != world['answer_ids']
            or len(world['source_mask']) != n or world['query_positions'] != list(range(p-1, n))
            or not 0 <= target_index < t or sorted(set(rows)) != rows
            or any(not 0 <= row < n for row in rows)):
        raise ValueError('World original, capture or vocabulary-target axis changed')
    _arrays(raw/'axes.npz', {name: np.asarray(world[name], dtype=bool if name=='source_mask' else np.int64)
                           for name in ('token_ids','answer_ids','source_mask','query_positions','capture_positions')})
    write_json(raw/'receivers.json', dict(receivers=world['capture_receivers'],
        all_original_query_rows=query_axis(world, world['answer_offsets'])))
    before = dict(counts)
    phase = []
    try:
        with torch.inference_mode(), ExitStack() as stack:
            precision = dict(inference_mode=torch.is_inference_mode_enabled(),
                grad_enabled=torch.is_grad_enabled(),autocast_cpu=torch.is_autocast_enabled('cpu'),
                autocast_cuda=torch.is_autocast_enabled('cuda'),
                cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                weight_parameter_dtypes=sorted({str(v.dtype) for v in observer.native.parameters()}),
                rotary_inverse_frequency_dtype=str(observer.native.model.rotary_emb.inv_freq.dtype))
            if (not precision['inference_mode'] or precision['grad_enabled'] or precision['autocast_cpu']
                    or precision['autocast_cuda'] or precision['cuda_matmul_tf32'] or precision['cudnn_tf32']
                    or precision['weight_parameter_dtypes']!=['torch.bfloat16']
                    or precision['rotary_inverse_frequency_dtype']!='torch.float32'):
                raise RuntimeError('Admitted native FP32 precision context changed')
            for layer in range(len(observer.layers)):
                traces[layer] = stack.enter_context(capture_rows(observer, rows, layer=layer))
            counts['model_calls'] += 1
            write_json(output/'model-attempt.json', dict(actual_counts=dict(counts),
                full_tokens=n, automatic_retry=False, precision_flags=precision))
            native = observer.hidden(world['token_ids'], use_cache=True)
        if (native.last_hidden_state.dtype != torch.float32
                or tuple(native.last_hidden_state.shape) != (1,n,observer.native.config.hidden_size)
                or not torch.isfinite(native.last_hidden_state).all()):
            raise RuntimeError('Complete native hidden axis/precision changed')
        selected = native.last_hidden_state[:, p-1:p-1+t, :].clone()
        _arrays(raw/'original-final-hidden.npz', {'hidden':selected[0].detach().cpu().numpy()})
        # Save one layer's cache at a time; never hold a private whole CPU cache.
        pairs = native.past_key_values.to_legacy_cache()
        if len(pairs) != len(observer.layers):
            raise RuntimeError('Native cache is missing a model layer')
        for layer, pair in enumerate(pairs):
            shape = (1, observer.head_layout(layer)[1], n, observer.head_layout(layer)[2])
            if len(pair)!=2 or any(value.dtype!=torch.float32 or tuple(value.shape)!=shape
                                  or not torch.isfinite(value).all() for value in pair):
                raise RuntimeError('Native KV dtype/GQA/full-input axis changed')
            _arrays(raw/f'kv-{layer:02d}.npz', {name:value.detach().cpu().numpy()
                for name,value in zip(('key','value'),pair,strict=True)})
        del pairs, pair
        # Each independent replay is compared with the same forward's pre-O AV.
        attention = output/'attention'
        attention.mkdir()
        for layer in range(len(observer.layers)):
            trace = traces[layer]
            _arrays(raw/f'trace-{layer:02d}.npz', {name:value.numpy() for name,value in trace.items()})
            diagnostic = attention_diagnostic(observer, trace, native.past_key_values, rows, layer=layer)
            _arrays(attention/f'layer-{layer:02d}.npz', {name:value for name,value in diagnostic.items()
                                                      if name not in {'key','value'}})
            _arrays(attention/f'profile-{layer:02d}.npz', _profiles(diagnostic, world))
            phase.append(dict(layer=layer, physical_heads=diagnostic['attention'].shape[0],
                receivers=len(rows), keys=n, av_max_error=diagnostic['av_max_error'],
                av_max_error_over_tolerance=diagnostic['av_max_error_over_tolerance']))
            del diagnostic, trace
            traces.pop(layer)
        # Native KV and final full hidden die BEFORE loading the large head.
        native = None
        allocator_cleanup(observer)
        counts['head_calls'] += 1
        write_json(output/'head-attempt.json', dict(actual_counts=dict(counts),
            rows=t, vocabulary=observer.native.config.vocab_size, unchunked=True, automatic_retry=False))
        logits = observer.logits(selected)[0].detach().cpu()
        _arrays(output/'output.npz', {'logits':logits.numpy()})
        if (logits.dtype!=torch.float32 or tuple(logits.shape)!=(t,observer.native.config.vocab_size)
                or not torch.isfinite(logits).all()):
            raise RuntimeError('Full original-target vocabulary output changed')
        endpoints = token_endpoints(logits, world['answer_ids'], world['answer_offsets'])
        axis = query_axis(world, world['answer_offsets'])
        for endpoint, row in zip(endpoints, axis[:-1], strict=True):
            endpoint['receiver_target_axis'] = row
        logp = logits[target_index].log_softmax(-1)
        values = {name:dict(token_id=token,logit=float(logits[target_index,token]),
                           logp=float(logp[token]),probability=float(logp[token].exp()))
                  for name,token in candidates.items()}
        result = dict(candidates=values, original_token_endpoints=endpoints,
            all_layers_AV_qualified=True, AV_policy=AV_POLICY, phase=phase,
            actual_counts=dict(counts), counts_before_world=before,
            hook_cleanup=_hook_counts(observer.native)==hooks,
            qualification='read-only capture and same-forward AV; no fresh paired no-op',
            historical_BF16_probability_identity='not_a_numerical_prerequisite_or_same_run_claim',
            resources=_resources(observer.device), new_detector_scores=0, auroc=None,
            precision_flags=precision,original_generator_causality=False)
        del logits, logp, selected
        allocator_cleanup(observer)
        if not result['hook_cleanup'] or not observer.resource_report()['all_parameters_meta']:
            raise RuntimeError('World hook/residency cleanup failed')
        write_json(output/'result.json', result)
        _seal(attention)
        _seal(raw)
        _seal(output)
        return result
    except BaseException as error:
        # Partial Q/cos/AV captures are retained even if a native forward fails.
        for layer, trace in traces.items():
            path = raw/f'trace-{layer:02d}.npz'
            if not path.exists():
                _arrays(path, {name:value.numpy() for name,value in trace.items()})
        write_json(output/'failure.json', dict(type=type(error).__name__,message=str(error),
            actual_counts=dict(counts), counts_before_world=before,
            automatic_retry=False, resources=_resources(observer.device)))
        _seal(raw)
        _seal(output)
        raise
    finally:
        native = None
        traces.clear()


def _load_inputs(prepared, primitive, license_path, audit_path):
    license_, audit = read_json(license_path), read_json(audit_path)
    if file_hash(license_path)!=LICENSE_SHA256 or file_hash(audit_path)!=AUDIT_SHA256:
        raise ValueError('Independent seven-day source license changed')
    bindings = {str(license_path):file_hash(license_path), str(audit_path):file_hash(audit_path)}
    for name, key in [('inputs.jsonl','prepared_inputs'),('manifest.json','prepared_manifest')]:
        path = prepared/name
        expected = license_['provenance'][key]['sha256']
        if file_hash(path)!=expected:
            raise ValueError('Original inference input identity changed')
        bindings[str(path)] = expected
    if read_json(prepared/'manifest.json').get('inputs_sha256')!=bindings[str(prepared/'inputs.jsonl')]:
        raise ValueError('Prepared manifest does not bind the original inputs')
    with (prepared/'inputs.jsonl').open(encoding='utf-8') as stream:
        sample = Sample(**json.loads(next(stream)))
    if (sample.id!='5670' or sample.source_id!='13601' or sample.task!='Data2txt'
            or sample.split!='test' or text_hash(sample.response)!=audit['case']['response_sha256']):
        raise ValueError('The prescribed original inference case changed')
    relative = ('sources/000026.json','responses/000176/response.json',
                'responses/000176/with_source.npz')
    paths = [primitive/name for name in relative]
    for path in paths:
        if audit['input_sha256'].get(str(path))!=file_hash(path):
            raise ValueError('Original primitive file identity differs from the frozen audit')
        bindings[str(path)] = file_hash(path)
    source, response = read_json(paths[0]), read_json(paths[1])
    with np.load(paths[2],allow_pickle=False) as saved:
        if not np.array_equal(saved['token_id'],np.asarray(response['answer_ids'],dtype=np.int64)):
            raise ValueError('Original scalar/response target IDs changed')
    if response['text']!=sample.response:
        raise ValueError('Original saved answer text differs')
    p, t = len(source['prompt_with_source']), len(response['answer_ids'])
    axes = dict(token_ids=source['prompt_with_source']+response['answer_ids'],
        answer_ids=response['answer_ids'],answer_offsets=response['offsets'],
        source_mask=source['source_mask']+[False]*t,query_positions=list(range(p-1,p+t)))
    return sample, axes, license_, bindings


def _reuse_units(root, previous_stage, prepared, assets, identity):
    """Reuse only the two sealed prefix worlds from the known progress-I/O failure."""
    if root is None:
        if previous_stage is not None:
            raise ValueError('Prior stage source is meaningful only with an explicit reused root')
        return {}, {}
    if previous_stage is None:
        raise ValueError('Explicit prior installed observation source is required for reuse')
    root, previous_stage = Path(root).resolve(), Path(previous_stage).resolve()
    verify_seal(root,required=('plan.json','eligibility.json','failure.json'))
    prior, failed = read_json(root/'plan.json'), read_json(root/'failure.json')
    previous_code = prior['source_runtime']['source_sha256']
    if (failed.get('type')!='FileExistsError' or 'progress.json' not in failed.get('message','')
            or failed.get('actual_counts')!={'model_calls':2,'head_calls':2}
            or prior['assets']!=assets or read_json(root/'eligibility.json')!=prepared
            or file_hash(previous_stage)!=previous_code['experiments/relation_scope_observation.py']):
        raise ValueError('Prior attempt is not the exact fixed two-world progress failure')
    ignored = {'experiments/relation_scope_observation.py','experiments/run.py'}
    current_code = identity['source_sha256']
    if ({k:v for k,v in previous_code.items() if k not in ignored}
            !={k:v for k,v in current_code.items() if k not in ignored}):
        raise ValueError('A reused numerical preparation/observation dependency changed')
    def function_ast(path, name):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
        if len(functions)!=1:
            raise ValueError('A unique unchanged observation function is required')
        return ast.dump(functions[0],include_attributes=False)
    if any(function_ast(previous_stage,name)!=function_ast(Path(__file__),name)
           for name in ('observe_world','_profiles')):
        raise ValueError('Reused world observation arithmetic changed')
    observations, receipts = {}, {}
    for name in WORLD_ORDER:
        unit = root/name
        if name not in ('native','S_minus'):
            if unit.exists():
                raise ValueError('Prior attempt contains an unexpected later world')
            continue
        verify_seal(unit,required=('result.json','output.npz'))
        verify_seal(unit/'raw')
        verify_seal(unit/'attention')
        value = read_json(unit/'result.json')
        if (value.get('all_layers_AV_qualified') is not True or value.get('hook_cleanup') is not True
                or len(value.get('original_token_endpoints',[]))!=len(prepared['answer_ids'])
                or [v['token_id'] for v in value['original_token_endpoints']]!=prepared['answer_ids']):
            raise ValueError('Reused original-world result is incomplete or unqualified')
        observations[name] = value
        receipts[name] = dict(directory=str(unit),seal_sha256=file_hash(unit/'seal.json'),
            original_stage_source_sha256=file_hash(previous_stage),model_recomputed=False)
    return observations, receipts


def observe_worlds(observer, prepared, output, *, reused=None, counts=None):
    """Complete the fixed population with immutable per-world progress receipts."""
    counts = counts if counts is not None else {'model_calls':0,'head_calls':0}
    if counts!={'model_calls':0,'head_calls':0}:
        raise ValueError('A new continuation starts with zero new attempts')
    observations = dict(reused or {})
    for name in WORLD_ORDER:
        if name not in observations:
            observations[name] = observe_world(observer,prepared['worlds'][name],prepared['candidate_ids'],
                output/name,counts=counts,target_index=prepared['target_answer_index'])
        write_json(output/f'progress-{name}.json',dict(completed=[n for n in WORLD_ORDER if n in observations],
            actual_counts=dict(counts),reused_worlds=list(reused or {}),
            planned_population=prepared['world_order'],automatic_retry=False))
    return observations, counts


def run(args):
    """Claim before invocation; one finite attempt with no automatic fallback."""
    import torch
    from transformers import AutoTokenizer
    started = time.perf_counter()
    names = ('prepared','primitive_root','capture_root','model','asset_manifest',
             'head_qualification','source_license','probability_audit','admission',
             'amendment','resource_review','output')
    paths = {name:Path(getattr(args,name)).resolve() for name in names}
    output, model = paths['output'], paths['model']
    if output.exists():
        raise FileExistsError('Existing observation attempt is never overwritten/restarted')
    for key, expected in [('admission',ADMISSION_SHA256),('source_license',LICENSE_SHA256),
                          ('resource_review',RESOURCE_SHA256),('amendment',AMENDMENT_SHA256)]:
        if file_hash(paths[key])!=expected:
            raise ValueError('Frozen independent admission/resource/license changed')
    if shutil.disk_usage(output.parent).free < 3*1024**3:
        raise MemoryError('Less than the frozen 3GiB output reserve is free')
    assets = verify_assets(paths['asset_manifest'],model,
        verification=getattr(args,'asset_verification',None),state=getattr(args,'asset_state',None))
    identity = capture_identity(paths['capture_root'],model/'config.json')
    prerequisite = head_prerequisite(paths['head_qualification'],assets,identity)
    sample, axes, license_, bindings = _load_inputs(paths['prepared'],paths['primitive_root'],
        paths['source_license'],paths['probability_audit'])
    bindings.update({str(paths[key]):file_hash(paths[key]) for key in
                     ('admission','amendment','resource_review','asset_manifest')})
    source_identity = research_identity()
    limits = ResourceLimits(max_sequence_tokens=1152)
    output.mkdir(exist_ok=False)
    counts = {'model_calls':0,'head_calls':0}
    observer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model,use_fast=True,local_files_only=True,trust_remote_code=False)
        prepared = prepare_worlds(sample,tokenizer,axes,license_,maximum=1152)
        write_json(output/'eligibility.json',prepared)
        del tokenizer
        observations, reused = _reuse_units(getattr(args,'reuse',None),getattr(args,'reuse_stage_source',None),
            prepared,assets,source_identity)
        if reused:
            for receipt in reused.values():
                path = Path(receipt['directory'])/'seal.json'
                bindings[str(path)] = file_hash(path)
            bindings[str(Path(args.reuse_stage_source).resolve())] = file_hash(Path(args.reuse_stage_source))
        planned = {name:5-len(reused) for name in ('model_calls','head_calls')}
        write_json(output/'plan.json',dict(schema='h017-five-world-scope-plan-v1',
            source_runtime=source_identity,input_bindings=bindings,assets=assets,
            observer_capture_identity=identity,head_prerequisite=prerequisite,
            resource_limits=limits.__dict__,planned_counts=planned,reused_units=reused,
            total_population_counts=prepared['budget'],
            primary='Splus minus each of Mon/Tue/Sat at originaltarget55 z00-z30',
            no_new_detector=True,target_annotations_read=False))
        store = SafeTensorStore(model,index_sha256=assets['files']['model.safetensors.index.json']['sha256'],
            expected_files=assets['files'])
        observer = StreamedObserver(store,read_json(model/'config.json'),config_sha256=assets['files']['config.json']['sha256'],
            device=getattr(args,'device','cuda:0'),limits=limits)
        observations, counts = observe_worlds(observer,prepared,output,reused=observations,counts=counts)
        if counts!=planned:
            raise RuntimeError('Actual original forward/head count differs from the frozen budget')
        evaluation = scientific_summary(observations)
        result = dict(evaluation,actual_counts=counts,reused_units=reused,
            total_evidence_counts=prepared['budget'],elapsed_seconds=time.perf_counter()-started,
            resource_report=observer.resource_report(),source_reading_closure_queries_observed=True,
            source_closure_semantic_encoding_or_use='not_inferred',
            target_annotation_reads=0,detector_evaluation=False)
        write_json(output/'result.json',result)
        if any(file_hash(Path(path))!=sha for path,sha in bindings.items()):
            raise RuntimeError('A frozen dependency changed during the observation')
        completed_identity(source_identity)
        _seal(output)
        return result
    except BaseException as error:
        write_json(output/'failure.json',dict(type=type(error).__name__,message=str(error),
            actual_counts=dict(counts),elapsed_seconds=time.perf_counter()-started,automatic_retry=False))
        _seal(output)
        raise
    finally:
        if observer is not None:
            observer.close()
