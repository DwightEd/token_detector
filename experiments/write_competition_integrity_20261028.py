"""Independent CPU integrity recomputation for the frozen write-competition run."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
from functools import lru_cache
import hashlib
import json
import urllib.request

import numpy as np
from scipy.special import roots_legendre, logsumexp

T=Path('.aris/traces/experiment-audit/20261028_write_competition')

R=Path('outputs/write_competition_20261028')
B=Path('outputs/source_write_20261028')
C=Counter();E={};H={}
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for v in iter(lambda:f.read(1048576),b''):h.update(v)
    return h.hexdigest()
def same(a,b,key,tol=3e-11):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape,(key,a.shape,b.shape)
    d=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.
    E[key]=max(E.get(key,0.),d)
    assert d<=tol,(key,d,tol)
    C[key]+=1
def seal(p):
    for n,h in read(p/'seal.json').items():
        assert sha(p/n)==h,(p,n)
        C['seal_entries']+=1
def identity(source,base):
    for n,h in source['source_sha256'].items():
        package,file=n.split('/',1);p=base/Path(package.replace('.','/'))/file
        assert sha(p)==h,(p,h)
        H[str(p)]=h
        C['source_files']+=1
@lru_cache(None)
def quad(n):
    x,w=roots_legendre((n+2)//2)
    return np.log((x+1)/2),np.log((1-x)/2),np.log(w/2)
def beta(d,answer,unit):
    out=np.empty(len(d))
    cuts=np.r_[0,np.flatnonzero((answer[1:]!=answer[:-1])|(unit[1:]!=unit[:-1]))+1,len(d)]
    for a,b in zip(cuts[:-1],cuts[1:]):
        v=d[a:b];lt,lc,lw=quad(len(v))
        f=np.logaddexp(lc[:,None],lt[:,None]+v)
        # Exclude the own likelihood factor using disjoint prefix/suffix sums.
        other=np.cumsum(f,axis=1)-f+np.cumsum(f[:,::-1],axis=1)[:,::-1]-f
        out[a:b]=v+logsumexp(lw[:,None]+lt[:,None]+other,axis=0)-logsumexp(lw[:,None]+lc[:,None]+other,axis=0)
        assert np.max(np.abs(out[a:b]-v))<=np.log(len(v))+5e-10
        C['independent_beta_span_integrals']+=1
        C['independent_beta_token_integrals']+=len(v)
    return out
def metrics(y,s,W):
    order=np.argsort(-s,kind='stable')
    y=y[order];s=s[order]
    first=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    out=[]
    for lo in range(0,len(W),32):
        w=W[lo:lo+32,order]
        p=np.add.reduceat(w*y,first,axis=1)
        n=np.add.reduceat(w*(~y),first,axis=1)
        cp,cn=np.cumsum(p,axis=1),np.cumsum(n,axis=1)
        P,N=p.sum(1),n.sum(1)
        with np.errstate(divide='ignore',invalid='ignore'):
            auc=(p*(N[:,None]-cn+n/2)).sum(1)/(P*N)
            precision=np.divide(cp,cp+cn,out=np.zeros_like(cp,dtype=float),where=cp+cn>0)
            ap=(p*precision).sum(1)/P
        out.append(np.column_stack([auc,ap]))
    return np.concatenate(out)
def point(y,s):
    v=metrics(y,s,np.ones((1,len(y))))[0]
    sel=np.argsort(-s,kind='stable')[:int(np.ceil(len(y)*.05))]
    tp=int(y[sel].sum())
    return {'tokens':len(y),'errors':int(y.sum()),'auroc':float(v[0]) if np.isfinite(v[0]) else None,
            'ap':float(v[1]) if np.isfinite(v[1]) else None,
            'top5pct':{'tp':tp,'fp':len(sel)-tp,'fn':int(y.sum())-tp}}
def boot(y,scores,groups,contrasts):
    unique,index=np.unique(groups,return_inverse=True)
    draws=np.random.default_rng(17).integers(0,len(unique),(1000,len(unique)))
    W=np.asarray([np.bincount(x,minlength=len(unique)) for x in draws])[:,index]
    ms={k:metrics(y,scores[k],W) for k in {x for pair in contrasts.values() for x in pair}}
    res={}
    for name,(a,b) in contrasts.items():
        r={'seed':17,'repetitions':1000,'unit':'source_id'}
        for j,k in enumerate(['auroc','ap']):
            d=ms[b][:,j]-ms[a][:,j];d=d[np.isfinite(d)]
            r[k]={'defined':len(d),'degenerate':1000-len(d),
                  'interval95':np.quantile(d,[.025,.975]).tolist() if len(d) else None}
        res[name]=r
    return res


def run(trace):
    global T
    T=Path(trace)
    T.mkdir(parents=True,exist_ok=False)
    plan=read(R/'plan.json');bp=read(B/'plan.json');D=Path(plan['parent']);dp=read(D/'plan.json')
    original=read(D/'inputs.json');cache=Path(original['cache'])
    complete=read(R/'score_complete.json');col=read(R/'collection.json');bc=read(B/'collection.json')
    assert plan['source']==complete['source']==col['source']==read(R/'prelaunch.json')['source']
    assert len(plan['source']['source_sha256'])==56 and len(bp['source']['source_sha256'])==55
    identity(plan['source'],Path('.'));identity(bp['source'],B/'installed_clean')
    assert col['observation_source']==bc['source']==bp['source']
    source_diff={'added':sorted(set(plan['source']['source_sha256'])-set(bp['source']['source_sha256'])),
                 'changed':sorted(k for k,h in bp['source']['source_sha256'].items()
                                  if plan['source']['source_sha256'].get(k)!=h)}
    assert source_diff=={'added':['experiments/source_write_integrity_20261028.py'],
                         'changed':['experiments/source_write_run.py']},source_diff
    assert plan['parent']==bp['parent'] and plan['reuse']['path']==str(B.resolve())
    assert plan['reuse']['seal_sha256']==sha(B/'seal.json')
    assert plan['reuse']['complete_sha256']==sha(B/'score_complete.json')
    assert plan['parent_seal_sha256']==sha(D/'seal.json')
    assert plan['parent_complete_sha256']==sha(D/'score_complete.json')
    assert plan['protocol_sha256']==sha(Path(plan['protocol']))
    assert col['reused'] is True and col['new_model_calls']==0 and col['label_reads']==0
    assert col['original_collection_sha256']==sha(B/'collection.json')
    assert complete['state_seal_sha256']==sha(R/'seal.json')
    assert set(complete['prediction_seals'])=={'old48','additional144'}
    for root in [R,B,D]:seal(root);seal(root/'case_readouts')
    for cohort,h in complete['prediction_seals'].items():
        out=R/f'scores_{cohort}';seal(out)
        assert sha(out/'seal.json')==h
        op=read(out/'plan.json')
        assert op['source']==plan['source'] and read(out/'complete.json')['source']==plan['source']
        assert op['new_observation_seal_sha256']==complete['state_seal_sha256']
        assert op['systems']==plan['systems'] and op['contrasts']==plan['contrasts']
        seal(B/f'scores_{cohort}')
    assert complete['case_seal_sha256']==sha(R/'case_readouts/seal.json')
    expected_contrasts={'versus_direct':['direct_pool_dependency','competition_pool_dependency'],
    'versus_full':['matched_full_pool_dependency','competition_pool_dependency'],
    'versus_r4':['r4_punct','competition_pool_dependency'],'point_contrast':['direct_gap','competition_gap']}
    assert plan['contrasts']==expected_contrasts
    assert len(plan['systems'])==19 and plan['systems']==bp['systems']+['competition_gap','competition_pool_dependency','competition_pool_punct']
    for n,h in original['input_hashes'].items():
        assert sha(cache/n)==h
        C['primitive_hashes']+=1
    obs={r['id']:r for r in col['records']};bobs={r['id']:r for r in bc['records']}
    assert len(obs)==len(col['records'])==len(original['records'])==196
    native={};responses={}
    for e in original['records']:
        r=e['record'];o=obs[r['id']];bo=bobs[r['id']]
        assert Path(o['file'])==(B/bo['file']).resolve()
        assert o['sha256']==bo['sha256']==sha(Path(o['file']))
        C['reused_native_hashes']+=1;H[o['file']]=o['sha256']
        resp=read(cache/r['directory']/'response.json');responses[r['id']]=resp
        with np.load(o['file'],allow_pickle=False) as z:
            assert np.array_equal(z['token_id'],resp['answer_ids'])
            native[r['id']]=z['direct_gap'].astype(np.float64)-z['history_gap'].astype(np.float64)
        if C['reused_native_hashes']%40==0:print('NATIVE_HASHES',C['reused_native_hashes'],flush=True)
    # Label bytes are only touched after all saved score/state/case seals.
    annfile=Path(dp['dataset'])/'response.jsonl';annhash=sha(annfile)
    url='https://raw.githubusercontent.com/ParticleMedia/RAGTruth/c103204b9ce28d6bbad859304bf30de72b8ed8fe/dataset/response.jsonl'
    h=hashlib.sha256();nb=0
    with urllib.request.urlopen(url,timeout=60) as response:
        while True:
            block=response.read(1048576)
            if not block:break
            h.update(block);nb+=len(block)
    assert h.hexdigest()==annhash
    wanted=set(obs);anns={}
    with annfile.open() as f:
        for line in f:
            row=json.loads(line)
            if str(row['id']) in wanted:anns[str(row['id'])]=row
    assert set(anns)==wanted
    results={};cohort_sources={};case_results={}
    for cohort,info in dp['cohorts'].items():
        source=Path(info['source']);seal(source)
        ip=read(source/'inputs.json');records=ip['records'];records_col=read(source/'collection.json')['records']
        cp=read(source/'plan.json');assert cp['dataset']==dp['dataset']
        assert sha(source/'inputs.json')==cp['inputs_sha256']
        tasks={};cohort_sources[cohort]={r['source_id'] for r in records}
        for task in ['QA','Summary','Data2txt']:
            parts=[]
            for ai,r in enumerate(records):
                if r['task']!=task:continue
                assert r['partition'] in ['fit','dev']
                resp=responses[r['id']];ids=np.asarray(resp['answer_ids']);offs=np.asarray(resp['offsets'])
                target=np.flatnonzero((offs[:,1]>offs[:,0])&~np.isin(ids,resp['special_ids']))
                ann=anns[r['id']]
                assert ann['response']==resp['text'] and str(ann['source_id'])==str(r['source_id'])
                y=np.array([any(offs[t,0]<x['end'] and offs[t,1]>x['start'] for x in ann['labels']) for t in target],bool)
                count=len(target)
                row={'answer':np.full(count,ai),'target':target,'token':ids[target],
                     'dev':np.full(count,r['partition']=='dev'),'source':np.full(count,r['source_id']),
                     'y':y,'gap':native[r['id']][target]}
                for method,key in [('dependency','direct_mdl'),('punct','punct')]:
                    row[method]=np.searchsorted(records_col[ai]['cuts'][key][1:],target,side='right')
                parts.append(row)
            rows={k:np.concatenate([x[k] for x in parts]) for k in parts[0]}
            dev=rows['dev'];y=rows['y'][dev];groups=rows['source'][dev]
            vals={'competition_gap':rows['gap']}
            for method in ['dependency','punct']:
                vals['competition_pool_'+method]=beta(rows['gap'],rows['answer'],rows[method])
            scores={};m={}
            for system in plan['systems']:
                file=R/f'scores_{cohort}'/f'{task}_{system}.npz'
                with np.load(file,allow_pickle=False) as z:saved=dict(z)
                for field,key in [('answer_index','answer'),('target','target'),('token_id','token')]:
                    assert np.array_equal(saved[field],rows[key][dev]),(cohort,task,system,field)
                    C['original_axis_arrays']+=1
                assert np.isfinite(saved['score']).all()
                if system in bp['systems']:
                    assert sha(file)==sha(B/f'scores_{cohort}'/file.name)
                    C['byte_copied_parent_scores']+=1
                else:
                    same(saved['score'],vals[system][dev],system,0 if system=='competition_gap' else 3e-11)
                    C['new_DEV_score_rows']+=len(y)
                scores[system]=saved['score'];m[system]=point(y,saved['score']);H[str(file)]=sha(file)
            b=boot(y,scores,groups,plan['contrasts'])
            tasks[task]={'DEV_answers':len(set(rows['answer'][dev])),'DEV_sources':len(set(groups)),
                         'DEV_tokens':len(y),'DEV_errors':int(y.sum()),'metrics':m,'paired_differences':b}
            print('TASK_RECOMPUTED',cohort,task,len(y),int(y.sum()),flush=True)
        results[cohort]=tasks
    assert not cohort_sources['old48']&cohort_sources['additional144']
    cases=read(R/'case_readouts/scores.json')['cases'];oldcases={x['id']:x for x in read(B/'case_readouts/scores.json')['cases']}
    cut_records={r['id']:r for r in read(D/'collection.json')['records']}
    assert {x['id'] for x in cases}=={'10988','2390','3180','3182'}
    for case in cases:
        old=oldcases[case['id']];target=np.asarray(case['target'])
        for k in ['source_id','response','offsets','target','token_id']:assert case[k]==old[k]
        v=native[case['id']][target];same(case['scores']['competition_gap'],v,'case_raw',0)
        assert set(case['scores'])=={'competition_gap','competition_pool_dependency','competition_pool_punct'}
        for method,key in [('dependency','direct_mdl'),('punct','punct')]:
            units=np.searchsorted(cut_records[case['id']]['cuts'][key][1:],target,side='right')
            same(case['scores']['competition_pool_'+method],beta(v,np.zeros(len(v)),units),'case_beta_'+method)
        C['case_tokens']+=len(v);C['cases']+=1
        case_results[case['id']]={'tokens':len(v),'character_axis_unchanged':True,'all_three_scores_recomputed':True}
    for vector in [np.zeros(4),np.array([0.]),np.array([4.,-2.,0.,1.5]),np.array([-2.,3.])]:
        # Compare quadrature to canonical pooler for arithmetic edge cases only.
        from myproject.pooling import cavity_scores
        same(cavity_scores(vector)['score'],beta(vector,np.zeros(len(vector)),np.zeros(len(vector))),'synthetic_beta')
    for f in [R/'plan.json',R/'collection.json',R/'seal.json',R/'score_complete.json',R/'case_readouts/scores.json',annfile]:
        H[str(f)]=sha(f)
    res={'generated_at':datetime.now(timezone.utc).isoformat(),'CPU_only':True,'production_metric_imports':False,
         'production_pooler_used_for_real_data':False,'counts':dict(C),'max_errors':E,
         'source_snapshots':{'current_entries':56,'parent_entries':55,'difference':source_diff,
                             'parent_snapshot':'outputs/source_write_20261028/installed_clean'},
         'annotation_provenance':{'upstream_url':url,'bytes':nb,'local_sha256':annhash,'upstream_sha256':h.hexdigest()},
         'results':results,'cases':case_results,'audited_input_hashes':H,
         'declared_skill_route_status':'REVIEW_UNAVAILABLE','review_independence':'same-family','acceptance_status':'provisional'}
    with (T/'002-numerical.json').open('x') as f:json.dump(res,f,indent=2)
    print(json.dumps({'counts':dict(C),'max_errors':E,'source_difference':source_diff},indent=2),flush=True)


def main():
    import argparse
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--trace',type=Path,required=True,help='A new directory; existing outputs are never overwritten')
    run(parser.parse_args().trace)


if __name__=='__main__':main()
