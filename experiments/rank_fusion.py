"""Seal fixed rank fusion of explicitly aligned score caches on CPU."""

from pathlib import Path
import re

import numpy as np

from . import data
from .provenance import completed_identity, research_identity
from .research.score_fusion import equal_rank_fusion


AXES=('sample_id','source_id','response_sha256','token_index','token_id','start','end')
RULE={'method':'r4_v3_equal_rank_v1',
      'ranking_scope':'complete_task_all_scored_tokens',
      'include_unknown_in_scoring':True,'r4_weight':.5}


def _text(value):
    if value.dtype.kind not in 'US':
        raise ValueError('Nonempty UTF-8 text identities required')
    try:
        value=np.char.decode(value,'utf-8') if value.dtype.kind=='S' else value
    except UnicodeDecodeError as error:
        raise ValueError('Nonempty UTF-8 text identities required') from error
    if np.any(value==''):
        raise ValueError('Nonempty UTF-8 text identities required')
    return value


def _validate_axes(result, shape):
    for key in AXES:
        value=result[key]
        if value.shape != shape:
            raise ValueError('Identity axes must match every scored token')
        if key in ['sample_id','source_id','response_sha256']:
            result[key]=_text(value)
        elif value.dtype.kind not in 'iu' or np.any(value<0):
            raise ValueError('Nonnegative integer token/offset axes required')
    if any(re.fullmatch(r'[0-9a-f]{64}',digest) is None
           for digest in set(result['response_sha256'].tolist())):
        raise ValueError('Response SHA256 identities must be 64 lowercase hexadecimal characters')
    if np.any(result['end']<=result['start']):
        raise ValueError('Scored tokens need positive character spans')
    if len(set(zip(result['sample_id'].tolist(),result['token_index'].tolist())))!=shape[0]:
        raise ValueError('Duplicate original token identity')
    response_identity={}
    for sample,source,digest in zip(result['sample_id'],result['source_id'],
                                    result['response_sha256'],strict=True):
        if response_identity.setdefault(sample,(source,digest))!=(source,digest):
            raise ValueError('A response changes source or text identity')


def _load(path, method):
    # Only score/identity fields are read, even if unrelated fields are present.
    with np.load(path, allow_pickle=False) as saved:
        result={key:saved[key] for key in (*AXES,'score')}
        for key in ['method','observer_id']:
            value=saved[key]
            if value.shape != ():
                raise ValueError('Scalar text method/observer identity required')
            result[key]=_text(value).item()
    if result['method'] != method:
        raise ValueError('Unexpected method in source score cache')
    scores=result['score']
    if (scores.ndim != 1 or not len(scores) or scores.dtype.kind not in 'fiu'
            or not np.isfinite(scores).all()):
        raise ValueError('Nonempty finite score vector required')
    _validate_axes(result,scores.shape)
    return result


def _load_population(path):
    with np.load(path,allow_pickle=False) as saved:
        if set(saved.files)!=set(AXES):
            raise ValueError('Population manifest must contain only the original token axes')
        result={key:saved[key] for key in AXES}
    if result['sample_id'].ndim!=1 or not len(result['sample_id']):
        raise ValueError('Nonempty one-dimensional population manifest required')
    _validate_axes(result,result['sample_id'].shape)
    return result


def score(r4, witness, protocol, output):
    """Verify identities, fuse without labels, and seal the result separately."""
    r4,witness,protocol,output=map(lambda p:Path(p).resolve(),(r4,witness,protocol,output))
    base=(Path.cwd()/'outputs').resolve()
    if output==base or not output.is_relative_to(base):
        raise ValueError('Fusion outputs must be under current-project outputs/<run-id>')
    bindings={str(path):data.file_hash(path) for path in [r4,witness,protocol]}
    declared=data.read_json(protocol)
    if (not isinstance(declared,dict) or any(declared.get(k)!=v for k,v in RULE.items())
            or not isinstance(declared.get('task'),str) or not declared['task']):
        raise ValueError('Protocol must freeze the exact equal-rank rule and task')
    if (not isinstance(declared.get('population_manifest'),str) or not declared['population_manifest']
            or not isinstance(declared.get('population_sha256'),str)):
        raise ValueError('Protocol must bind an independent label-free population manifest')
    population_path=Path(declared['population_manifest'])
    if not population_path.is_absolute():
        population_path=protocol.parent/population_path
    population_path=population_path.resolve()
    bindings[str(population_path)]=data.file_hash(population_path)
    if bindings[str(population_path)]!=declared['population_sha256']:
        raise ValueError('Frozen population manifest changed')
    identity=research_identity()
    population=_load_population(population_path)
    left,right=_load(r4,'r4'),_load(witness,'witness')
    if (not left['observer_id'] or left['observer_id']!=right['observer_id']
            or declared.get('observer_id')!=left['observer_id']):
        raise ValueError('Observer identities differ')
    if any(not np.array_equal(left[key],right[key]) for key in AXES):
        raise ValueError('R4 and V3 original token axes differ; do not silently join or drop')
    if any(not np.array_equal(left[key],population[key]) for key in AXES):
        raise ValueError('Score coverage differs from the frozen population manifest')
    fused=equal_rank_fusion(left['score'],right['score'])
    if any(data.file_hash(Path(path))!=digest for path,digest in bindings.items()):
        raise ValueError('Input scores, frozen protocol or population changed')
    completed_identity(identity)
    output.mkdir(parents=True,exist_ok=False)
    with (output/'scores.npz').open('xb') as stream:
        np.savez_compressed(stream,**{key:left[key] for key in AXES},score=fused,
                            r4_score=left['score'],witness_score=right['score'],
                            method=np.array(RULE['method']),observer_id=np.array(left['observer_id']))
    data.write_json(output/'run.json',{'rule':RULE,'task':declared['task'],
        'observer_id':left['observer_id'],'tokens_scored':len(fused),
        'responses_scored':len(set(left['sample_id'].tolist())),
        'population_manifest':str(population_path),'population_sha256':declared['population_sha256'],
        'input_sha256':bindings,'source':identity,'annotation_reads':0,
        'status':'scores_sealed_not_evaluated','metrics_computed':False,
        'population_rule':'all original tokens in the frozen label-free population manifest; strata only at evaluation'})
    data.write_json(output/'seal.json',{name:data.file_hash(output/name)
                                       for name in ['scores.npz','run.json']})
