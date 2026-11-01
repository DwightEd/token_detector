"""Independent CPU integrity recomputation. Production scorers/evaluators are not imported."""
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import hashlib, json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp, roots_legendre

R=Path('outputs/source_write_20261028')
T=Path('.aris/traces/experiment-audit/20261028_source_write')
COUNTS=Counter(); ERRORS={}; HASHES={}
def read(p): return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def verify(p):
    for n,h in read(p/'seal.json').items():
        assert sha(p/n)==h,(p,n)
        COUNTS['seal_entries']+=1
def close(k,a,b,tol=5e-11):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape,(k,a.shape,b.shape)
    error=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.
    ERRORS[k]=max(ERRORS.get(k,0.),error)
    assert error<=tol,(k,error,tol)
    COUNTS[k]+=1
def bounds(answer,unit):
    return np.r_[0,np.flatnonzero((answer[1:]!=answer[:-1])|(unit[1:]!=unit[:-1]))+1,len(answer)]
@lru_cache(None)
def quadrature(n):
    x,w=roots_legendre((n+2)//2)
    return np.log((x+1)/2),np.log((1-x)/2),np.log(w/2)
def pooled(gap,answer,unit):
    out=np.empty(len(gap)); cuts=bounds(answer,unit)
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        d=gap[lo:hi]; lt,lc,lw=quadrature(len(d))
        f=np.logaddexp(lc[:,None],lt[:,None]+d)
        other=np.cumsum(f,axis=1)-f
        other+=np.cumsum(f[:,::-1],axis=1)[:,::-1]-f
        out[lo:hi]=d+logsumexp(lw[:,None]+lt[:,None]+other,axis=0)-logsumexp(lw[:,None]+lc[:,None]+other,axis=0)
        violation=float(max(0.,np.max(np.abs(out[lo:hi]-d))-np.log(len(d))))
        ERRORS['uniform_Beta_correction_bound_excess']=max(ERRORS.get('uniform_Beta_correction_bound_excess',0.),violation)
        assert violation<=2e-10,(len(d),violation)
        COUNTS['independent_Beta_integrals']+=len(d)
    return out
def rank_vector(y,s,weights):
    order=np.argsort(-s,kind='stable'); z=s[order]; y=y[order]
    starts=np.r_[0,np.flatnonzero(z[1:]!=z[:-1])+1]
    out=[]
    for first in range(0,len(weights),50):
        w=weights[first:first+50,order]
        pos=np.add.reduceat(w*y,starts,axis=1)
        neg=np.add.reduceat(w*(~y),starts,axis=1)
        cp,cn=pos.cumsum(1),neg.cumsum(1)
        np_,nn_=pos.sum(1),neg.sum(1)
        with np.errstate(invalid='ignore',divide='ignore'):
            auc=(pos*(nn_[:,None]-cn+.5*neg)).sum(1)/(np_*nn_)
            prec=np.divide(cp,cp+cn,out=np.zeros_like(cp,dtype=float),where=(cp+cn)>0)
            ap=(pos*prec).sum(1)/np_
        out.append(np.column_stack([auc,ap]))
    return np.concatenate(out)
def point(y,s):
    a=rank_vector(y,s,np.ones((1,len(y))))[0]
    alarm=np.argsort(-s,kind='stable')[:int(np.ceil(.05*len(s)))]
    tp=int(y[alarm].sum())
    return {'tokens':len(y),'errors':int(y.sum()),'auroc':float(a[0]) if np.isfinite(a[0]) else None,'ap':float(a[1]) if np.isfinite(a[1]) else None,
            'top5pct':{'tp':tp,'fp':len(alarm)-tp,'fn':int(y.sum())-tp}}
def metric_close(a,b):
    for k in ['tokens','errors','top5pct']:assert a[k]==b[k],(k,a[k],b[k])
    for k in ['auroc','ap']:
        if a[k] is None:assert b[k] is None
        else:close('metric_'+k,a[k],b[k],2e-14)
def bootstrap(y,scores,groups,contrasts):
    unique,index=np.unique(groups,return_inverse=True)
    draws=np.random.default_rng(17).integers(0,len(unique),(1000,len(unique)))
    source_weights=np.array([np.bincount(d,minlength=len(unique)) for d in draws])
    weights=source_weights[:,index]
    needed={x for p in contrasts.values() for x in p}
    vals={name:rank_vector(y,scores[name],weights) for name in needed}
    result={}
    for name,(baseline,candidate) in contrasts.items():
        r={'seed':17,'repetitions':1000,'unit':'source_id'}
        for j,k in enumerate(['auroc','ap']):
            d=vals[candidate][:,j]-vals[baseline][:,j];d=d[np.isfinite(d)]
            r[k]={'defined':len(d),'degenerate':1000-len(d),
                  'interval95':np.quantile(d,[.025,.975]).tolist() if len(d) else None}
        result[name]=r
    return result

def run():
    plan=read(R/'plan.json'); parent=Path(plan['parent']); pp=read(parent/'plan.json')
    original=read(parent/'inputs.json'); cache=Path(original['cache'])
    for name,wanted in plan['source']['source_sha256'].items():
        package,file=name.split('/',1);path=Path(package.replace('.','/'))/file
        HASHES[str(path)]=sha(path)
        assert HASHES[str(path)]==wanted
    for name,wanted in original['input_hashes'].items():
        assert sha(cache/name)==wanted
        COUNTS['primitive_hashes_rechecked']+=1
    c=read(R/'collection.json');complete=read(R/'score_complete.json')
    assert c['source']==plan['source']==complete['source']
    assert c['counts']=={k:read(T/'001-inputs.json')['expected_'+k] for k in ['backbone_calls','processed_tokens']}
    assert c['weight_hash_scope']=='config/tokenizer/index, not weight shards'
    for n,h in c['model_files_sha256'].items():assert sha(Path(pp['model'])/n)==h
    assert c['model_files_sha256']==read(parent/'collection.json')['model_files_sha256']
    assert complete['state_seal_sha256']==sha(R/'seal.json')
    verify(R);verify(parent);verify(R/'case_readouts')
    assert complete['case_seal_sha256']==sha(R/'case_readouts'/'seal.json')
    assert set(complete['prediction_seals'])=={'old48','additional144'}
    for cohort,h in complete['prediction_seals'].items():
        out=R/f'scores_{cohort}';verify(out)
        assert sha(out/'seal.json')==h
        op=read(out/'plan.json')
        assert op['new_observation_seal_sha256']==complete['state_seal_sha256']
        assert op['systems']==plan['systems'] and op['contrasts']==plan['contrasts']
    # Read official annotations only after all score/case seals and complete gate.
    annpath=Path(pp['dataset'])/'response.jsonl'
    for cohort in pp['cohorts']:
        assert read(parent/f'scores_{cohort}'/'evaluation.json')['annotation_sha256']==sha(annpath)
    wanted={e['record']['id'] for e in original['records']}
    annotations={}
    with annpath.open() as f:
        for line in f:
            row=json.loads(line)
            if str(row['id']) in wanted:annotations[str(row['id'])]=row
    native={};responses={};summary={};scalars=['native_logp','direct_gap','history_gap','source_gradient','source_projection','smooth_logp','smooth_minus_source_logp','matched_full_gap']
    obs={r['id']:r for r in c['records']}
    parent_obs={x['id']:x for x in read(parent/'collection.json')['records']}
    assert len(obs)==len(c['records'])==len(original['records'])==196
    for e in original['records']:
        r=e['record'];file=R/obs[r['id']]['file']
        HASHES[str(file)]=sha(file)
        parent_file=parent/parent_obs[r['id']]['file']
        HASHES[str(parent_file)]=sha(parent_file)
        assert HASHES[str(file)]==obs[r['id']]['sha256']
        assert HASHES[str(parent_file)]==parent_obs[r['id']]['sha256']
        COUNTS['parent_native_hashes']+=1
        COUNTS['native_hashes']+=1
        response=read(cache/r['directory']/'response.json');responses[r['id']]=response
        with np.load(file,allow_pickle=False) as raw:
            a={k:raw[k] for k in raw.files}
        n=len(response['answer_ids'])
        assert np.array_equal(a['token_id'],response['answer_ids'])
        assert set(a)==set(scalars+['token_id','source_write','history_write','other_write','native_attention','mlp_write','embedding','residual','rounding'])
        for k,v in a.items():
            assert v.shape==((n,4096) if k in ['source_write','history_write','other_write','native_attention','mlp_write','embedding','residual','rounding'] else (n,)),(r['id'],k,v.shape)
            assert np.isfinite(v).all(),(r['id'],k)
        assert all(np.all(a[k]<=0) for k in ['native_logp','smooth_logp','smooth_minus_source_logp'])
        for k in ['embedding','residual']:
            assert a[k].dtype==np.float32 and not np.any(a[k].view(np.uint32)&65535)
            COUNTS['BF16_exact_native_elements']+=a[k].size
        close('stored_rounding',((a['residual']-a['embedding'])-a['native_attention'])-a['mlp_write'],a['rounding'],0)
        close('residual_reconstruction',a['residual'],a['embedding']+a['native_attention']+a['mlp_write']+a['rounding'],2e-4)
        close('rounding_diagnostic',np.abs(a['rounding']).max(),obs[r['id']]['diagnostics']['residual_add_max_abs'],0)
        assert obs[r['id']]['diagnostics']['norm_replay_max_abs']==0
        partition=np.abs(a['source_write']+a['history_write']+a['other_write']-a['native_attention']).max()
        d=obs[r['id']]['diagnostics']['partition_max_abs']
        assert partition<=32*d+2e-3,(r['id'],partition,d)
        for k,v in {'summed_partition_vs_native_max':partition,'native_residual_add_max':np.abs(a['rounding']).max(),
                    'smooth_native_logp_offset_max':np.abs(a['smooth_logp']-a['native_logp']).max(),
                    'smooth_finite_vs_gradient_max':np.abs(a['smooth_minus_source_logp']-a['smooth_logp']-a['source_gradient']).max(),
                    'native_finite_vs_gradient_max':np.abs(a['direct_gap']-a['source_gradient']).max()}.items():
            summary[k]=max(summary.get(k,0.),float(v))
        assert np.array_equal(a['history_write'][0],np.zeros(4096))
        with np.load(cache/r['directory']/'without_source.npz',allow_pickle=False) as old:
            close('matched_full_gap',old['full'].astype(float)-a['native_logp'],a['matched_full_gap'],0)
        native[r['id']]={k:a[k] for k in scalars}
        native[r['id']]['history_support']=-a['history_gap']
        native[r['id']]['native_nll']=-a['native_logp']
        COUNTS['original_answer_tokens']+=n
        if COUNTS['native_hashes']%25==0:print('AUDITED_NATIVE',COUNTS['native_hashes'],flush=True)
    results={}
    copied=['native_gap','native_pool_dependency','native_pool_punct','r4_punct','frozen_r4']
    points=['direct_gap','history_gap','history_support','source_gradient','source_projection','native_nll','matched_full_gap']
    for cohort,cohort_info in pp['cohorts'].items():
        out=R/f'scores_{cohort}';source=Path(cohort_info['source'])
        records=read(source/'inputs.json')['records'];collection=read(source/'collection.json')['records']
        evaluation=read(out/'evaluation.json')
        assert evaluation['annotation_sha256']==sha(annpath)
        assert evaluation['scope']==plan['scope']
        tasks={}
        for task in ['QA','Summary','Data2txt']:
            parts=[]
            for answer,r in enumerate(records):
                if r['task']!=task:continue
                response=responses[r['id']];offsets=np.asarray(response['offsets']);ids=np.asarray(response['answer_ids'])
                target=np.flatnonzero((offsets[:,1]>offsets[:,0])&~np.isin(ids,response['special_ids']))
                ann=annotations[r['id']]
                assert ann['response']==response['text'] and str(ann['source_id'])==str(r['source_id'])
                y=np.array([any(offsets[t,0]<span['end'] and offsets[t,1]>span['start'] for span in ann['labels']) for t in target],bool)
                m=len(target)
                row={'answer':np.full(m,answer),'target':target,'token':ids[target],
                     'dev':np.full(m,r['partition']=='dev'),'source':np.full(m,r['source_id']),'y':y}
                for k in points:row[k]=native[r['id']][k][target]
                with np.load(parent/parent_obs[r['id']]['file'],allow_pickle=False) as oldraw:
                    row['native_gap']=oldraw['absent_logp'][target].astype(float)-oldraw['present_logp'][target]
                for method,key in [('punct','punct'),('dependency','direct_mdl')]:
                    row[method]=np.searchsorted(collection[answer]['cuts'][key][1:],target,side='right')
                parts.append(row)
            rows={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
            dev=rows['dev'];y=rows['y'][dev];groups=rows['source'][dev]
            for prefix,observation in [('direct','direct_gap'),('matched_full','matched_full_gap'),('native','native_gap')]:
                for method in ['dependency','punct']:
                    rows[prefix+'_pool_'+method]=pooled(rows[observation],rows['answer'],rows[method])
            scores={};metrics={}
            for system in plan['systems']:
                file=out/f'{task}_{system}.npz'
                with np.load(file,allow_pickle=False) as z:saved={k:z[k] for k in z.files}
                for k,axis in [('answer_index','answer'),('target','target'),('token_id','token')]:
                    assert np.array_equal(saved[k],rows[axis][dev]),(cohort,task,system,k)
                    COUNTS['original_axis_arrays']+=1
                assert np.isfinite(saved['score']).all()
                if system in copied:
                    assert sha(file)==sha(parent/f'scores_{cohort}'/file.name)
                    COUNTS['baseline_byte_copies']+=1
                if system in rows:
                    close('score_'+system,saved['score'],rows[system][dev])
                    COUNTS['recomputed_DEV_score_rows']+=len(y)
                scores[system]=saved['score'];metrics[system]=point(y,saved['score'])
                metric_close(metrics[system],evaluation['tasks'][task]['metrics'][system])
                COUNTS['metric_systems']+=1
                HASHES[str(file)]=sha(file)
            b=bootstrap(y,scores,groups,plan['contrasts'])
            for name,result in b.items():
                expected=evaluation['tasks'][task]['paired_differences'][name]
                for k in ['seed','repetitions','unit']:assert result[k]==expected[k]
                for k in ['auroc','ap']:
                    for f in ['defined','degenerate']:assert result[k][f]==expected[k][f]
                    if result[k]['interval95'] is None:assert expected[k]['interval95'] is None
                    else:close('bootstrap_'+k,result[k]['interval95'],expected[k]['interval95'],2e-14)
                COUNTS['paired_source_bootstrap_comparisons']+=1
            assert evaluation['tasks'][task]['dev_sources']==len(set(groups))
            tasks[task]={'DEV_answers':len(set(rows['answer'][dev])),'DEV_sources':len(set(groups)),
                         'DEV_tokens':len(y),'DEV_errors':int(y.sum()),'metrics':metrics,'paired_differences':b}
            print('AUDITED_TASK',cohort,task,len(y),int(y.sum()),flush=True)
        results[cohort]=tasks
    cases=read(R/'case_readouts'/'scores.json')['cases']
    oldcases={x['id']:x for x in read(parent/'case_readouts'/'scores.json')['cases']}
    for case in cases:
        old=oldcases[case['id']];target=np.asarray(case['target'])
        for k in ['source_id','response','offsets','target','token_id']:assert case[k]==old[k]
        for k in points:close('case_'+k,case['scores'][k],native[case['id']][k][target],0)
        for method,key in [('dependency','direct_mdl'),('punct','punct')]:
            units=np.searchsorted(parent_obs[case['id']]['cuts'][key][1:],target,side='right')
            d=pooled(native[case['id']]['direct_gap'][target],np.zeros(len(target)),units)
            close('case_direct_pool_'+method,case['scores']['direct_pool_'+method],d)
        COUNTS['case_outputs']+=1
    for file in [R/'plan.json',R/'collection.json',R/'seal.json',R/'score_complete.json',R/'case_readouts/scores.json',annpath]:
        HASHES[str(file)]=sha(file)
    result={'generated_at':datetime.now(timezone.utc).isoformat(),'counts':dict(COUNTS),'max_errors':ERRORS,
            'native_diagnostics':summary,'results':results,'audited_input_hashes':HASHES,
            'declared_skill_route_status':'REVIEW_UNAVAILABLE','review_independence':'same-family',
            'acceptance_status':'provisional','CPU_only':True,'production_scorer_imports':False}
    with (T/'003-numerical.json').open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'counts':dict(COUNTS),'max_errors':ERRORS,'native_diagnostics':summary},indent=2))
if __name__=='__main__':run()

