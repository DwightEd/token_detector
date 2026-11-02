"""Summarize recovered R4/V3 metrics; no detector scoring or label selection."""
import csv
from decimal import Decimal
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/results'
TASKS=['QA','Summary','Data2txt','FinQA','TATQA']
DISPLAY={'QA':'RAGTruth QA','Summary':'RAGTruth Summary','Data2txt':'RAGTruth Data2txt',
         'FinQA':'FRED FinQA','TATQA':'FRED TATQA','HalluRAG_available':'HalluRAG 包含答案'}


def main():
    records=json.loads((OUT/'comparison.json').read_text(encoding='utf-8'))['records']
    evidence=json.loads((OUT/'hallurag_strata_evidence.json').read_text(encoding='utf-8'))
    def lookup(group,method,task):
        matches=[r for r in records if (r['group'],r['method'],r['task'])==(group,method,task)]
        assert len(matches)==1,(group,method,task)
        return matches[0]
    rows=[]
    for task in [*TASKS,'HalluRAG_available']:
        group='llama_historical' if task!='HalluRAG_available' else 'llama_hallurag_r4_historical'
        a=lookup(group,'R4',task)
        b=lookup('llama_historical' if task!='HalluRAG_available' else 'llama_hallurag',
                 'V3 witness',task)
        best=max(Decimal(a['auroc']),Decimal(b['auroc']))
        rows.append({'task':task,'r4_auroc':a['auroc'],'r4_ap':a['ap'],
                     'v3_auroc':b['auroc'],'v3_ap':b['ap'],
                     'delta_v3_minus_r4_auroc':str(Decimal(b['auroc'])-Decimal(a['auroc'])),
                     'r4_point_ge_0_8':Decimal(a['auroc'])>=Decimal('.8'),
                     'v3_point_ge_0_8':Decimal(b['auroc'])>=Decimal('.8'),
                     'descriptive_envelope_gap':str(max(Decimal('0'),Decimal('.8')-best)),
                     'comparison_kind':'descriptive; observation budgets/executions differ'})
    packed={'new_experiment':False,'selection':'original RAGTruth TEST; released answerable metadata for HalluRAG primary group; no label-based filtering',
            'observer':'Llama-3.1-8B-Instruct','target_auroc':'0.8',
            'r4_point_passes':sum(r['r4_point_ge_0_8'] for r in rows),
            'v3_point_passes':sum(r['v3_point_ge_0_8'] for r in rows),
            'descriptive_envelope_passes':sum(Decimal(r['descriptive_envelope_gap'])==0 for r in rows),
            'envelope_is_not_a_detector':True,'tasks':rows}
    (OUT/'r4_v3_task_summary.json').write_text(json.dumps(packed,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    s=io.StringIO(newline='');writer=csv.DictWriter(s,list(rows[0]),lineterminator='\n')
    writer.writeheader();writer.writerows(rows)
    (OUT/'r4_v3_task_summary.csv').write_text(s.getvalue(),encoding='utf-8-sig')
    table=['| 主任务 | R4 AUROC / AP | V3 AUROC / AP | 已有最好点值距0.8 |',
           '|---|---:|---:|---:|']
    for r in rows:
        table.append('| '+DISPLAY[r['task']]+' | '+format(float(r['r4_auroc']),'.4f')+'/'+format(float(r['r4_ap']),'.4f')+
                     ' | '+format(float(r['v3_auroc']),'.4f')+'/'+format(float(r['v3_ap']),'.4f')+
                     ' | '+('点值已达' if Decimal(r['descriptive_envelope_gap'])==0 else format(Decimal(r['descriptive_envelope_gap']),'.4f'))+' |')
    strata=['| HalluRAG 分组 | 回答数 | R4（Llama） | V3（Llama） | V3（Qwen dense） |',
            '|---|---:|---:|---:|---:|']
    views=[('all','pooled','全部',1782),
           ('answerable_target_available','target_available','包含答案／主组',891),
           ('unanswerable_target_unavailable','target_unavailable','不含答案／鲁棒性',891)]
    for key,view,title,n in views:
        r4=next(r for r in evidence['r4']['metrics'] if r['stratum']==key)
        a=next(r for r in evidence['v3']['rows'] if r['observer'].startswith('Llama') and r['view']==view)
        b=next(r for r in evidence['v3']['rows'] if r['observer'].startswith('Qwen') and r['view']==view)
        strata.append(f"| {title} | {n} | {float(r4['auroc_printed']):.4f}/{float(r4['ap_printed']):.4f} | {a['auroc']}/{a['ap']} | {b['auroc']}/{b['ap']} |")
    document=ROOT/'docs/source-reading/PERFORMANCE_REVIEW.md'
    text=document.read_text(encoding='utf-8')
    for name,body in [('TASK_TABLE','\n'.join(table)),('STRATA_TABLE','\n'.join(strata))]:
        start,end=f'<!-- {name}_START -->',f'<!-- {name}_END -->'
        left,rest=text.split(start,1);_,right=rest.split(end,1)
        text=left+start+'\n\n'+body+'\n\n'+end+right
    document.write_text(text,encoding='utf-8')
    assert len(rows)==6 and packed['r4_point_passes']==2 and packed['v3_point_passes']==3
    assert packed['descriptive_envelope_passes']==4
    print(json.dumps({k:v for k,v in packed.items() if k!='tasks'},ensure_ascii=True,indent=2))


if __name__=='__main__':
    main()
