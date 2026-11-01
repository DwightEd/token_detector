"""Frozen direct-write capture, shared-local readout, and official-label evaluation."""

import argparse
from pathlib import Path
import shutil
import time

import numpy as np

from experiments import data
from experiments.delta_local_run import evaluation_gate, validate
from experiments.fit_ablation import evaluate, verify_seal
from experiments.pooling_run import _unit_index
from experiments.provenance import completed_identity, research_identity
from myproject.pooling import grouped_scores
from experiments.span_pilot import load_batches, validated_inputs

COPIED = ['native_gap','native_pool_dependency','native_pool_punct','r4_punct','frozen_r4']
POINTS = ['direct_gap','history_gap','history_support','source_gradient',
          'source_projection','native_nll','matched_full_gap']
SYSTEMS = [*POINTS,'direct_pool_dependency','direct_pool_punct',
           'matched_full_pool_dependency','matched_full_pool_punct',*COPIED]
CONTRASTS = {
    'matched_new_observation':('matched_full_pool_dependency','direct_pool_dependency'),
    'new_observation':('native_pool_dependency','direct_pool_dependency'),
    'matched_punctuation':('native_pool_punct','direct_pool_punct'),
    'versus_r4':('r4_punct','direct_pool_dependency'),
    'direct_vs_full':('native_gap','direct_gap'),
    'gradient_vs_finite':('direct_gap','source_gradient'),
    'uncentered_vs_gradient':('source_gradient','source_projection'),
}
COMPETITION = ['competition_gap','competition_pool_dependency','competition_pool_punct']
COMPETITION_CONTRASTS = {
    'versus_direct':('direct_pool_dependency','competition_pool_dependency'),
    'versus_full':('matched_full_pool_dependency','competition_pool_dependency'),
    'versus_r4':('r4_punct','competition_pool_dependency'),
    'point_contrast':('direct_gap','competition_gap'),
}


def prepare(root, parent, protocol, reuse=None):
    evaluation_gate(parent)
    if reuse is not None:
        verify_completed_scores(reuse)
        if data.read_json(reuse/'plan.json')['parent']!=str(parent.resolve()):
            raise ValueError('Reused native measurements have a different original parent')
    root.mkdir(parents=True,exist_ok=False)
    data.write_json(root/'plan.json',{'parent':str(parent.resolve()),
        'parent_seal_sha256':data.file_hash(parent/'seal.json'),
        'parent_complete_sha256':data.file_hash(parent/'score_complete.json'),
        'protocol':str(protocol.resolve()),'protocol_sha256':data.file_hash(protocol),
        'source':research_identity(),'systems':SYSTEMS+(COMPETITION if reuse is not None else []),
        'contrasts':COMPETITION_CONTRASTS if reuse is not None else CONTRASTS,
        'reuse':None if reuse is None else {'path':str(reuse.resolve()),
            'seal_sha256':data.file_hash(reuse/'seal.json'),
            'complete_sha256':data.file_hash(reuse/'score_complete.json')},
        'scope':'exploratory historical exposed FIT/DEV and four selected cases; not untouched TEST',
        'label_reads':0})
    if reuse is not None:
        original=data.read_json(reuse/'collection.json')
        records=[{**r,'file':str((reuse/r['file']).resolve())} for r in original['records']]
        data.write_json(root/'collection.json',{'records':records,'reused':True,'new_model_calls':0,
            'observation_source':original['source'],'source':research_identity(),
            'original_collection_sha256':data.file_hash(reuse/'collection.json'),'label_reads':0})
        data.write_json(root/'seal.json',{n:data.file_hash(root/n) for n in ['plan.json','collection.json']})


def inputs(root):
    plan=data.read_json(root/'plan.json');parent=Path(plan['parent'])
    if research_identity()!=plan['source']:
        raise ValueError('Source/runtime changed since prepare')
    if (data.file_hash(parent/'seal.json')!=plan['parent_seal_sha256']
            or data.file_hash(parent/'score_complete.json')!=plan['parent_complete_sha256']
            or data.file_hash(Path(plan['protocol']))!=plan['protocol_sha256']):
        raise ValueError('Frozen parent/protocol changed')
    evaluation_gate(parent)
    if plan.get('reuse') is not None:
        reuse=plan['reuse'];base=Path(reuse['path']);verify_completed_scores(base)
        if (data.file_hash(base/'seal.json')!=reuse['seal_sha256']
                or data.file_hash(base/'score_complete.json')!=reuse['complete_sha256']):
            raise ValueError('Reused score/state bindings changed')
    parent_plan,original=validate(parent)
    return plan,parent,parent_plan,original


def collect(root):
    import torch
    from experiments.source_write_observe import write_trace
    from myproject.model import load_model

    plan,parent,parent_plan,original=inputs(root)
    if plan.get('reuse') is not None:raise ValueError('Reused runs must not collect again')
    identity=research_identity();destination=root/'observations';destination.mkdir(exist_ok=False)
    model_path=Path(parent_plan['model'])
    prior_model=data.read_json(parent/'collection.json')['model_files_sha256']
    model_hashes={name:data.file_hash(model_path/name) for name in prior_model}
    if model_hashes!=prior_model:raise ValueError('Model metadata differs from parent')
    torch.manual_seed(42);model=load_model(model_path,'cuda:0')
    counts={'backbone_calls':0,'processed_tokens':0}

    def count(_module,_args,kwargs):
        counts['backbone_calls']+=1;counts['processed_tokens']+=kwargs['input_ids'].numel()

    handle=model.model.register_forward_pre_hook(count,with_kwargs=True)
    torch.cuda.reset_peak_memory_stats();start=time.monotonic();records=[]
    try:
        for entry in original['records']:
            r=entry['record'];cache=Path(original['cache'])
            response=data.read_json(cache/r['directory']/'response.json')
            source=data.read_json(cache/r['source_file'])
            arrays,diagnostics=write_trace(model,source['prompt_with_source'],response['answer_ids'],source['source_mask'])
            arrays['token_id']=np.asarray(response['answer_ids'])
            if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('Nonfinite observation')
            if diagnostics['norm_replay_max_abs']!=0:raise ValueError('Native norm replay failed')
            with np.load(cache/r['directory']/'without_source.npz') as raw:
                arrays['matched_full_gap']=raw['full'].astype(float)-arrays['native_logp']
            file=destination/f'{r["id"]}.npz'
            with file.open('xb') as stream:np.savez_compressed(stream,**arrays)
            records.append({'id':r['id'],'file':str(file.relative_to(root)),
                            'sha256':data.file_hash(file),'diagnostics':diagnostics})
            print('COLLECTED',len(records),r['id'],flush=True)
    finally:handle.remove()
    torch.cuda.synchronize();inputs(root)
    data.write_json(root/'collection.json',{'records':records,'source':completed_identity(identity),
        'model_files_sha256':model_hashes,'weight_hash_scope':'config/tokenizer/index, not weight shards',
        'counts':counts,'seconds_including_io':time.monotonic()-start,
        'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'label_reads':0})
    data.write_json(root/'seal.json',{n:data.file_hash(root/n) for n in ['plan.json','collection.json']})


def readouts(raw, competition=False):
    if competition:
        return {'competition_gap':raw['direct_gap'].astype(float)-raw['history_gap']}
    return {key:(-raw['history_gap'] if key=='history_support' else
                 -raw['native_logp'] if key=='native_nll' else raw[key]) for key in POINTS}


def score(root):
    verify_seal(root);plan,parent,parent_plan,original=inputs(root);identity=research_identity()
    competition=plan.get('reuse') is not None
    copied=SYSTEMS if competition else COPIED
    copy_from=Path(plan['reuse']['path']) if competition else parent
    points=['competition_gap'] if competition else POINTS
    observed={r['id']:r for r in data.read_json(root/'collection.json')['records']}
    for r in observed.values():
        if data.file_hash(root/r['file'])!=r['sha256']:raise ValueError('Native observations changed')
    for name,cohort in parent_plan['cohorts'].items():
        source=Path(cohort['source']);_,cohort_inputs=validated_inputs(source)
        batches=load_batches(source,['punct']);collection=data.read_json(source/'collection.json')
        output=root/f'scores_{name}';output.mkdir(exist_ok=False)
        data.write_json(output/'plan.json',{'source_run':str(source),
            'source_seal_sha256':cohort['source_seal_sha256'],'new_observation_run':str(root.resolve()),
            'new_observation_seal_sha256':data.file_hash(root/'seal.json'),
            'systems':plan['systems'],'contrasts':plan['contrasts'],'source':identity,'label_reads':0,
            'scope':data.read_json(root/'plan.json')['scope']})
        for task in data.TASKS:
            packed=batches[(task,'punct')];parts={k:[] for k in points}
            for answer in np.unique(packed['answer_index']):
                selected=packed['answer_index']==answer;target=packed['target'][selected]
                record=cohort_inputs['records'][answer]
                with np.load(root/observed[record['id']]['file']) as raw:
                    if not np.array_equal(raw['token_id'][target],packed['token_id'][selected]):
                        raise ValueError('Original token identities differ')
                    for key,value in readouts(raw,competition).items():parts[key].append(value[target])
            values={k:np.concatenate(v) for k,v in parts.items()}
            for method,key in [('dependency','direct_mdl'),('punct','punct')]:
                units=_unit_index(key,packed,collection)
                observations=[('competition','competition_gap')] if competition else [('direct','direct_gap'),('matched_full','matched_full_gap')]
                for prefix,observation in observations:
                    values[f'{prefix}_pool_{method}']=grouped_scores(values[observation],packed['answer_index'],units)['score']
            dev=packed['development']
            for system in plan['systems']:
                file=output/f'{task}_{system}.npz'
                if system in copied:
                    with (copy_from/f'scores_{name}'/file.name).open('rb') as src,file.open('xb') as dst:
                        shutil.copyfileobj(src,dst)
                else:
                    if not np.isfinite(values[system]).all():raise ValueError('Nonfinite score')
                    with file.open('xb') as stream:
                        np.savez_compressed(stream,score=values[system][dev],
                            **{k:packed[k][dev] for k in ['target','token_id','answer_index']})
            print('SCORED',name,task,flush=True)
        data.write_json(output/'complete.json',{'source':completed_identity(identity),'label_reads':0})
        data.write_json(output/'seal.json',{f.name:data.file_hash(f) for f in output.iterdir()})
    cases=[];old_cases={c['id']:c for c in data.read_json(parent/'case_readouts/scores.json')['cases']}
    parent_records={r['id']:r for r in data.read_json(parent/'collection.json')['records']}
    for entry in original['records']:
        if 'case' not in entry['uses']:continue
        record=entry['record'];old=old_cases[record['id']];target=np.asarray(old['target'])
        with np.load(root/observed[record['id']]['file']) as raw:
            values={k:v[target] for k,v in readouts(raw,competition).items()}
            for method,key in [('dependency','direct_mdl'),('punct','punct')]:
                units=np.searchsorted(parent_records[record['id']]['cuts'][key][1:],target,side='right')
                prefix='competition' if competition else 'direct'
                values[f'{prefix}_pool_{method}']=grouped_scores(values[f'{prefix}_gap'],np.zeros(len(target)),units)['score']
        cases.append({**{k:old[k] for k in ['id','source_id','response','offsets','target','token_id']},
                      'scores':{k:v.tolist() for k,v in values.items()}})
    destination=root/'case_readouts';destination.mkdir(exist_ok=False)
    data.write_json(destination/'scores.json',{'cases':cases,'label_reads':0})
    data.write_json(destination/'seal.json',{'scores.json':data.file_hash(destination/'scores.json')})
    for r in observed.values():
        if data.file_hash(root/r['file'])!=r['sha256']:raise ValueError('Observation changed while scoring')
    data.write_json(root/'score_complete.json',{'state_seal_sha256':data.file_hash(root/'seal.json'),
        'prediction_seals':{name:data.file_hash(root/f'scores_{name}'/'seal.json') for name in parent_plan['cohorts']},
        'case_seal_sha256':data.file_hash(destination/'seal.json'),'source':completed_identity(identity),'label_reads':0})


def verify_completed_scores(root):
    """Source-independent artifact gate; later CPU trials may reuse a frozen run."""
    complete=data.read_json(root/'score_complete.json');verify_seal(root)
    if data.file_hash(root/'seal.json')!=complete['state_seal_sha256']:raise ValueError('Native seal changed')
    if set(complete['prediction_seals'])!={'old48','additional144'}:raise ValueError('Incomplete cohorts')
    for cohort,expected in complete['prediction_seals'].items():
        output=root/f'scores_{cohort}';verify_seal(output)
        if data.file_hash(output/'seal.json')!=expected:raise ValueError('Prediction seal changed')
        if data.read_json(output/'plan.json')['new_observation_seal_sha256']!=complete['state_seal_sha256']:
            raise ValueError('Prediction/native binding changed')
    verify_seal(root/'case_readouts')
    if data.file_hash(root/'case_readouts'/'seal.json')!=complete['case_seal_sha256']:raise ValueError('Cases changed')


def evaluate_cohort(root,name):
    inputs(root);verify_completed_scores(root)
    evaluate(root/f'scores_{name}')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=['prepare','collect','score','evaluate'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--parent',type=Path)
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--reuse',type=Path,help='Reuse a completed native run for the fixed competition readout')
    parser.add_argument('--cohort',choices=['old48','additional144'])
    args=parser.parse_args()
    if args.stage=='prepare':
        if args.parent is None or args.protocol is None:parser.error('prepare requires parent/protocol')
        prepare(args.output,args.parent,args.protocol,args.reuse)
    elif args.stage=='collect':collect(args.output)
    elif args.stage=='score':score(args.output)
    else:
        if args.cohort is None:parser.error('evaluate requires cohort')
        evaluate_cohort(args.output,args.cohort)


if __name__=='__main__':main()
