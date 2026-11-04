"""Finite CPU head/carrier association assay on qualified saved graphs.

This stage preserves original prediction rows and reports pre-O mathematics.
It neither imports a model nor loads projections, labels, or detector scores.
"""
from itertools import combinations
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .data import file_hash, read_json, verify_seal, write_json
from .provenance import completed_identity, research_identity
from .relation_intervention import _seal


ARGUMENTS = ('graphs', 'samples', 'licence', 'qualification', 'protocol')
LIMITS = dict(layers=32, query_heads=32, kv_heads=8, heads_per_group=4,
              max_pairs=8, max_tuples=1_000_000, query_chunk=8,
              max_private_bytes=512*1024**2, max_staging_bytes=16*1024**2,
              max_output_bytes=2*1024**3, min_free_disk_bytes=5*1024**3,
              time_limit_seconds=600)
HEAD_PAIRS = tuple((4*g+i,4*g+j,g) for g in range(8) for i,j in combinations(range(4),2))
COUNTERS = dict(model_calls=0, head_calls=0, mlp_calls=0, tokenizer_calls=0,
                annotation_reads=0, risk_outputs=0)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _arguments(args):
    return {key: str(Path(getattr(args,key)).resolve()) for key in ARGUMENTS}


def _protocol(args):
    protocol = read_json(args.protocol)
    _require(protocol.get('schema') == 'h019-head-carrier-association-v1'
             and protocol.get('limits') == LIMITS
             and protocol.get('source_id') == '14375'
             and protocol.get('cases') == ['00012','00013']
             and protocol.get('counters') == COUNTERS
             and protocol.get('m2_admitted') is False
             and protocol.get('detector_admitted') is False,
             'Frozen finite pre-O protocol differs')
    bindings = protocol.get('scientific_bindings')
    _require(isinstance(bindings,dict) and len(bindings) >= 3,
             'Scientific proposal, minimal contract and independent review are required')
    for name,digest in bindings.items():
        _require(file_hash(Path(name)) == digest, 'Scientific admission changed: '+name)
    return protocol


def _new_output(args):
    output = Path(args.output).resolve()
    for key in ('graphs','samples','plan'):
        value = getattr(args,key,None)
        if value is not None:
            _require(not output.is_relative_to(Path(value).resolve()),
                     'Output must not mutate prerequisite tree: '+key)
    output.mkdir(parents=True,exist_ok=False)
    return output


def _private_bytes():
    """Committed private memory on Windows; conservative peak RSS elsewhere."""
    if os.name == 'nt':
        import ctypes
        class Counters(ctypes.Structure):
            _fields_ = [('size',ctypes.c_ulong),('faults',ctypes.c_ulong),
                        *[(name,ctypes.c_size_t) for name in ('peak_working','working',
                          'peak_paged','paged','peak_nonpaged','nonpaged',
                          'pagefile','peak_pagefile','private')]]
        value = Counters(); value.size = ctypes.sizeof(value)
        process = ctypes.windll.kernel32.GetCurrentProcess
        process.restype = ctypes.c_void_p
        memory = ctypes.windll.psapi.GetProcessMemoryInfo
        memory.argtypes = [ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong]
        if not memory(process(),ctypes.byref(value),value.size):
            raise OSError('CPU process memory accounting failed')
        return int(value.private)
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if os.sys.platform == 'darwin' else peak*1024)


def _check_resources(output, started, accounting):
    private = _private_bytes()
    accounting['max_observed_private_or_conservative_rss_bytes'] = max(
        accounting.get('max_observed_private_or_conservative_rss_bytes',0),private)
    _require(private+LIMITS['max_staging_bytes'] <= LIMITS['max_private_bytes'],
             'CPU private-memory limit, including staging reserve, exceeded')
    _require(time.monotonic()-started <= LIMITS['time_limit_seconds'], 'Finite CPU time limit exceeded')
    _require(shutil.disk_usage(output).free >= LIMITS['min_free_disk_bytes'],
             'CPU output disk reserve not available')


def _verify_files(frontend):
    """Reuse witnessed large-file hashes; never call a stat a new body hash."""
    for item in frontend['input_files']:
        path = Path(item['path']); stat = path.stat()
        _require(stat.st_size == item['bytes'] and stat.st_mtime_ns == item['mtime_ns'],
                 'Prepared input stat changed: '+str(path))
        if item.get('hash_mode') != 'qualified_manifest_not_rehashed':
            _require(file_hash(path) == item['sha256'], 'Prepared metadata changed: '+str(path))


def prepare(args):
    from .head_carrier_frontend import prepare_frontend
    output = _new_output(args)
    try:
        protocol, identity = _protocol(args), research_identity()
        frontend = prepare_frontend(args.graphs,args.samples,args.licence,args.qualification)
        _require(frontend['source_id'] == protocol['source_id'], 'Source identity differs')
        _require([case['case_id'] for case in frontend['cases']] == protocol['cases'],
                 'Finite graph population differs')
        plan = dict(schema='h019-head-carrier-execution-plan-v1',
            arguments=_arguments(args), protocol=protocol, source_runtime=identity,
            frontend=frontend, new_model_calls=0, annotation_reads=0,
            frontend_only=True, m1_numeric_tuples=0)
        write_json(output/'plan.json',plan)
        report = dict(completed=True,stage='M0',gate_passed=frontend['gate_passed'],
            source_runtime=completed_identity(identity),F=frontend['F'],Q=frontend['Q'],
            planned_tuples=frontend['tuple_count'],m1_numeric_tuples=0,
            actual_counts=COUNTERS,detector_validated=False,auroc=None)
        write_json(output/'result.json',report); _seal(output)
        return report
    except BaseException as error:
        write_json(output/'failure.json',dict(type=type(error).__name__,message=str(error),
            actual_counts=COUNTERS,automatic_retry=False,m1_numeric_tuples=0))
        _seal(output)
        raise


def _save_npz(path, payload, accounting):
    logical = sum(value.nbytes for value in payload.values())
    _require(logical <= LIMITS['max_staging_bytes'], 'Output staging exceeds finite bound')
    _require(accounting['array_output_bytes']+logical+65536 <=
             LIMITS['max_output_bytes']-LIMITS['max_staging_bytes'],
             'Output file would exceed the finite disk budget')
    with path.open('xb') as stream:
        np.savez(stream,**payload)
    accounting['array_output_bytes'] += path.stat().st_size
    _require(accounting['array_output_bytes'] <= LIMITS['max_output_bytes']-LIMITS['max_staging_bytes'],
             'Array output budget exceeded; metadata reserve retained')
    accounting['max_single_output_staging_bytes'] = max(
        accounting.get('max_single_output_staging_bytes',0),logical)


def _payload(result, rows, pair_index):
    batch = len(rows)*len(HEAD_PAIRS)
    payload = {key: np.asarray(value).reshape((batch,)+np.asarray(value).shape[2:])
               for key,value in result.items()
               if isinstance(value,np.ndarray) and key != 'value_difference'}
    payload.update(target=np.repeat([row['t'] for row in rows],48).astype(np.int32),
        query=np.repeat([row['q'] for row in rows],48).astype(np.int32),
        target_id=np.repeat([row['target_id'] for row in rows],48).astype(np.int32),
        head1=np.tile([h1 for h1,_,_ in HEAD_PAIRS],len(rows)).astype(np.int8),
        head2=np.tile([h2 for _,h2,_ in HEAD_PAIRS],len(rows)).astype(np.int8),
        kv_group=np.tile([g for _,_,g in HEAD_PAIRS],len(rows)).astype(np.int8),
        pair_index=np.full(batch,pair_index,dtype=np.int16))
    return payload


def _case(case, frontend, args, output, accounting, started):
    from .research.head_carrier_association import erase_head_carrier_association
    case_id = case['case_id']; root = Path(args.graphs)/case_id
    n,p,t = int(case['N']),int(case['P']),int(case['T'])
    _require(n == p+t-1, 'Fresh input-node/prediction axis differs')
    attention = np.load(root/'attention.npy',mmap_mode='r',allow_pickle=False)
    values = np.load(root/'values.npy',mmap_mode='r',allow_pickle=False)
    targets = np.load(root/'target_ids.npy',mmap_mode='r',allow_pickle=False)
    _require(attention.shape == (32,32,n,n) and attention.dtype == np.float32
             and values.shape == (32,n,8,128) and values.dtype == np.float32,
             'Qualified physical graph layout changed')
    rows = [row for row in case['all_targets'] if row['selected']]
    _require(all(row['q']==p+row['t']-1 and int(targets[row['t']])==row['target_id'] for row in rows),
             'Original chosen target or prechoice row changed')
    pairs = frontend['eligible_pairs']; case_output = output/('case-'+case_id)
    case_output.mkdir()
    write_json(case_output/'axis.json',case)
    before = accounting['completed_tuples']; zero_layer0 = True
    stats = {pair['pair_id']:dict(tuples=0,nonzero_concat_norm=0,chi_undefined=0,
                                max_concat_norm=0.,sum_concat_norm=0.) for pair in pairs}
    for layer in range(32):
        _check_resources(output,started,accounting)
        differences = []
        for pair in pairs:
            a,b = pair['positions']
            _require(0 <= a < p and 0 <= b < p and a != b, 'Carrier left the source prompt')
            differences.append(values[layer,a].astype(np.float64)-values[layer,b].astype(np.float64))
        _save_npz(case_output/f'layer-{layer:02d}-value-differences.npz',
                  {'value_difference':np.stack(differences)},accounting)
        for pair_index,pair in enumerate(pairs):
            positions = np.asarray(pair['positions'],dtype=np.int64)
            group = np.asarray([g for _,_,g in HEAD_PAIRS],dtype=np.int64)
            selected_v = values[layer,positions[None,:],group[:,None],:][None,...]
            for chunk_start in range(0,len(rows),LIMITS['query_chunk']):
                _check_resources(output,started,accounting)
                selected = rows[chunk_start:chunk_start+LIMITS['query_chunk']]
                queries = np.asarray([row['q'] for row in selected],dtype=np.int64)
                _require(bool((positions[None,:] <= queries[:,None]).all()), 'Carrier causality differs')
                blocks = np.empty((len(queries),48,2,2),dtype=np.float32)
                for index,(h1,h2,_) in enumerate(HEAD_PAIRS):
                    blocks[:,index,0,:] = attention[layer,h1,queries[:,None],positions[None,:]]
                    blocks[:,index,1,:] = attention[layer,h2,queries[:,None],positions[None,:]]
                accounting['attempted_tuples'] += len(queries)*48
                result = erase_head_carrier_association(blocks,selected_v,all_visible=True)
                payload = _payload(result,selected,pair_index)
                filename = f'layer-{layer:02d}-pair-{pair_index:02d}-chunk-{chunk_start:04d}.npz'
                _save_npz(case_output/filename,payload,accounting)
                _require(result['passed'] is True, 'Frozen F64 association numeric gate failed; raw chunk retained')
                norms = np.asarray(result['concat_norm'])
                if layer == 0:
                    zero_layer0 = zero_layer0 and bool((norms == 0).all())
                    _require(zero_layer0, 'Same-ID layer0 V negative control failed')
                counts = stats[pair['pair_id']]
                counts['tuples'] += norms.size
                counts['nonzero_concat_norm'] += int(np.count_nonzero(norms))
                counts['chi_undefined'] += int(np.count_nonzero(~result['chi_defined']))
                counts['max_concat_norm'] = max(counts['max_concat_norm'],float(norms.max()))
                counts['sum_concat_norm'] += float(norms.sum(dtype=np.float64))
                accounting['completed_tuples'] += len(queries)*48
        write_json(case_output/f'progress-layer-{layer:02d}.json',dict(
            completed_layers=layer+1,completed_tuples=accounting['completed_tuples']-before))
    report = dict(completed=True,case_id=case_id,original_targets=t,
        selected_original_queries=len(rows),source_pairs=len(pairs),
        independent_head_pairs=48,layers=32,
        completed_tuples=accounting['completed_tuples']-before,
        layer0_same_ID_pre_O_zero=zero_layer0,per_source_pair_descriptive_only=stats,
        W_O_loaded=False,output_projection_unknown=True,semantic_specificity='unknown',
        no_downstream_or_truth_claim=True,actual_counts=COUNTERS)
    write_json(case_output/'result.json',report); _seal(case_output)
    return report


def run(args):
    if args.prepare_only:
        return prepare(args)
    _require(args.plan is not None, 'Execution requires a sealed M0 plan')
    plan_root = Path(args.plan).resolve()
    verify_seal(plan_root,required=('plan.json','result.json'))
    plan,identity = read_json(plan_root/'plan.json'),research_identity()
    _require(plan.get('schema')=='h019-head-carrier-execution-plan-v1'
             and plan['arguments']==_arguments(args)
             and plan['source_runtime']==identity and plan['protocol']==_protocol(args),
             'Frozen source/runtime, argument or scientific identity differs')
    frontend = plan['frontend']
    _require(frontend['gate_passed'] is True and 1 <= frontend['F'] <= 8
             and 0 < frontend['tuple_count'] <= LIMITS['max_tuples'], 'M0 did not admit this finite run')
    _verify_files(frontend)
    output = _new_output(args); started = time.monotonic()
    accounting = dict(attempted_tuples=0,completed_tuples=0,array_output_bytes=0,
                      output_metadata_reserve_bytes=LIMITS['max_staging_bytes'])
    completed = {}
    try:
        _check_resources(output,started,accounting)
        write_json(output/'plan.json',plan)
        for case in frontend['cases']:
            completed[case['case_id']] = _case(case,frontend,args,output,accounting,started)
        _require(accounting['completed_tuples']==frontend['tuple_count'], 'Finite tuple coverage differs')
        _verify_files(frontend)
        report = dict(completed=True,stage='M1_pre_O_only',cases=completed,
            actual_counts=COUNTERS,accounting=accounting,
            source_runtime=completed_identity(identity),
            elapsed_seconds=time.monotonic()-started,independent_interventions_not_summed=True,
            W_O_loaded=False,M2='not_admitted_unknown',semantic_specificity='unknown',
            method_promotion=False,detector_validated=False,auroc=None,automatic_retry=False)
        write_json(output/'result.json',report); _seal(output)
        return report
    except BaseException as error:
        write_json(output/'failure.json',dict(type=type(error).__name__,message=str(error),
            actual_counts=COUNTERS,accounting=accounting,completed_cases=list(completed),
            elapsed_seconds=time.monotonic()-started,automatic_retry=False))
        _seal(output)
        raise
