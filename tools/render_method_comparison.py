"""Render recovered historical results. Does not score or select detectors."""
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/results'
TASKS = ['QA', 'Summary', 'Data2txt', 'FinQA', 'TATQA', 'HalluRAG']
PAIR = re.compile(r'(0\.\d+)/(0\.\d+)')
ALIASES = {'V3 integrated':'V3 witness', 'Integrated V3':'V3 witness',
           'Frozen R4':'R4', 'Matched':'Matched local', 'Symmetric':'Symmetric local',
           'No persistence (c = 0)':'V3 c=0', '−MMD only':'-MMD only'}


def clean_prefix(line):
    prefix = PAIR.split(line)[0]
    return re.sub(r'^\d+\s+', '', prefix).strip()


def main():
    evidence_path = OUT/'evidence.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    records = []
    def add(group, method, task, auc, ap, source, status='historical_reported', **extras):
        assert 0 <= float(auc) <= 1 and 0 <= float(ap) <= 1
        records.append(dict(group=group, method=method, task=task, auroc=auc, ap=ap,
                            source=source, status=status, **extras))

    for table in evidence['tables']:
        tid = table['table']
        for line in table['rows']:
            pairs = PAIR.findall(line)
            prefix = clean_prefix(line)
            source = table['id']
            if tid in (2, 3, 26, 35, 41):
                method = ALIASES.get(prefix, prefix)
                group = {2:'llama_historical',3:'qwen_dense',26:'qwen_dense',
                         35:'llama_hallurag',41:'qwen_cached_archive'}[tid]
                tasks = (TASKS[:5] if tid == 2 else
                         ['HalluRAG', 'HalluRAG_available', 'HalluRAG_unavailable'] if tid == 35 else TASKS)
                for task, (auc, ap) in zip(tasks, pairs, strict=True):
                    # Table26 repeats the exact rounded LUMINA row in Table3.
                    if tid == 26 and method == 'LUMINA':
                        old = next(r for r in records if (r['group'],r['method'],r['task']) ==
                                   (group,method,task))
                        assert (old['auroc'], old['ap']) == (auc,ap)
                        continue
                    add(group,method,task,auc,ap,source)
            elif tid == 38:
                match = re.fullmatch(r'(\S+)\s+(Integrated V3|LUMINA)\s+(Original|Span mean|Span centered)',prefix)
                assert match, prefix
                task,method,transform=match.groups()
                local = re.search(r'\s(0\.\d{4})\s*$',line).group(1)
                add('qwen_dense_deletions',ALIASES.get(method,method)+' / '+transform,
                    task,*pairs[0],source,local_auroc=local)
            elif tid == 37:
                task = prefix
                for method, pair in zip(['V3 witness','V3 span mean'],pairs[:2],strict=True):
                    add('llama_historical_deletions',method,task,*pair,source)
            elif tid == 17:
                # PDF text can append the next line number directly to AP.
                # Table17 explicitly prints four decimals, never that suffix.
                match = re.search(r'(QA|Summary|Data2txt)\s+Source-full (NLL|entropy)\s+\d+\s+(0\.\d{4})\s+(0\.\d{4})',line)
                assert match,line
                task,method,auc,ap=match.groups()
                add('llama_historical','Source '+method,task,auc,ap,source)

    for output in evidence['history_outputs']:
        for line in output['rows']:
            task = re.match(r'test_(\w+)',line).group(1)
            task = {'finqa':'FinQA','tatqa':'TATQA'}.get(task,task)
            for field in output['fields']:
                match = re.search(re.escape(repr(field))+r': (\([^)]*\))',line)
                assert match, (field,line)
                values=ast.literal_eval(match.group(1))
                method={'witness_score':'V3 witness','token_state_score':'Token state',
                        'adaptive_token_score':'Adaptive token','onset_score':'Onset'}[field]
                add('llama_later_candidates',method,task,
                    *(format(x,'.6f') for x in values),output['id'],
                    status='retained_historical_candidate' if field=='witness_score' else 'rejected_historical_candidate')
    # Two recovered tool outputs repeat the same V3 row. Verify, then deduplicate.
    unique={}
    for r in records:
        key=(r['group'],r['method'],r['task'])
        if key in unique:
            assert (unique[key]['auroc'],unique[key]['ap']) == (r['auroc'],r['ap'])
        else:
            unique[key]=r
    records=list(unique.values())
    ledger=(ROOT/'experiments/EXPERIMENTS.md').read_text(encoding='utf-8')
    permutation=re.findall(r'^\|(QA|Summary|Data2txt|FinQA|TATQA)\|'
                           r'(\.\d+)/(\.\d+)\|(\.\d+)/(\.\d+)\|(\.\d+)/(\.\d+)\|$',ledger,re.M)
    assert len(permutation)==5
    for task,c0auc,c0ap,clipauc,clipap,auc,ap in permutation:
        for method, values in [('V3 c=0',(c0auc,c0ap)),('Fixed clipping',(clipauc,clipap))]:
            old=next(r for r in records if (r['group'],r['method'],r['task']) ==
                     ('llama_historical',method,task))
            assert all(abs(float(old[k])-float(v))<=.000051
                       for k,v in zip(['auroc','ap'],values,strict=True))
        add('llama_historical','V3 span permutation',task,'0'+auc,'0'+ap,
            'experiments/EXPERIMENTS.md',status='historical_ablation')
    # Exact, already published ledger values; require every literal to occur there.
    later = {
        'Responsibility': [('QA','.820332','.171270'),('Summary','.722511','.079181'),
                           ('Data2txt','.707506','.115250'),('FinQA','.743079','.269941'),('TATQA','.785727','.381096')],
        'Conditional': [('FinQA','.718312','.218039'),('TATQA','.770173','.357203')],
    }
    for method,rows in later.items():
        for task,auc,ap in rows:
            assert auc+'/'+ap in ledger, (method,task)
            add('llama_later_candidates',method,task,'0'+auc,'0'+ap,
                'experiments/EXPERIMENTS.md',status='rejected_historical_candidate')
    for method, rows in {
        'Relational world_shared': [('FinQA','.693095','.248915'),('TATQA','.674423','.305032')],
        'Same-run SDPA shared control': [('FinQA','.729517','.260282'),('TATQA','.768258','.355036')],
    }.items():
        for task,auc,ap in rows:
            assert auc+'/'+ap in ledger
            add('llama_relational_archive',method,task,'0'+auc,'0'+ap,
                'experiments/EXPERIMENTS.md',status='rejected_historical_candidate'
                if method.startswith('Relational') else 'matched_historical_control')
    strata_path=OUT/'hallurag_strata_evidence.json'
    strata=json.loads(strata_path.read_text(encoding='utf-8'))
    task_names={'all':'HalluRAG','answerable_target_available':'HalluRAG_available',
                'unanswerable_target_unavailable':'HalluRAG_unavailable'}
    for r in strata['r4']['metrics']:
        add('llama_hallurag_r4_historical','R4',task_names[r['stratum']],
            r['auroc_printed'],r['ap_printed'],'r4_hallurag_history_record_5299')
    for r in strata['v3']['rows']:
        if r['observer'].startswith('Qwen') and r['view']!='pooled':
            task='HalluRAG_available' if r['view']=='target_available' else 'HalluRAG_unavailable'
            add('qwen_dense','V3 witness',task,r['auroc'],r['ap'],'paper_table_28')
    for r in records:
        r['observer'] = 'Qwen2.5-3B-Instruct' if r['group'].startswith('qwen') else 'Llama-3.1-8B-Instruct'
        r['label_type'] = ('financial_silver' if r['task'] in ['FinQA','TATQA'] else
                           'sentence_derived_automatic' if r['task'].startswith('HalluRAG') else 'human_span')
    # Cross-table check: same dense predictions, different printed precision.
    for r in records:
        if r['group']=='qwen_dense_deletions' and r['method'].endswith(' / Original'):
            base=next(x for x in records if (x['group'],x['method'],x['task']) ==
                      ('qwen_dense',r['method'].split(' / ')[0],r['task']))
            assert all(abs(float(r[k])-float(base[k])) <= .00055 for k in ['auroc','ap'])

    precision={2:4,3:3,26:3,28:4,35:4,38:4,41:3,37:4,17:4}
    for r in records:
        if r['source'].startswith('paper_table_'):
            digits=precision[int(r['source'].removeprefix('paper_table_'))]
            assert all(len(r[k].split('.')[1])==digits for k in ['auroc','ap']),r
    packed={'schema':1,'as_of':evidence['as_of'],'new_experiments':False,
            'evidence_sha256':hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            'hallurag_strata_evidence_sha256':hashlib.sha256(strata_path.read_bytes()).hexdigest(),
            'ledger_sha256':hashlib.sha256((ROOT/'experiments/EXPERIMENTS.md').read_bytes()).hexdigest(),
            'records':records}
    (OUT/'comparison.json').write_text(json.dumps(packed,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fields=['group','observer','method','task','auroc','ap','label_type','status','source','local_auroc']
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fields,lineterminator='\n')
    writer.writeheader();writer.writerows(records)
    (OUT/'comparison.csv').write_text(stream.getvalue(),encoding='utf-8-sig')

    def matrix(groups, tasks, methods=None):
        selected=[r for r in records if r['group'] in groups]
        methods=methods or list(dict.fromkeys(r['method'] for r in selected))
        rows=['| 方法 | '+' | '.join(tasks)+' |','|---|'+'---:|'*len(tasks)]
        for method in methods:
            cells=[]
            for task in tasks:
                hits=[r for r in selected if r['method']==method and r['task']==task]
                assert len(hits)<=1,(groups,method,task)
                cells.append(hits[0]['auroc']+'/'+hits[0]['ap'] if hits else '—')
            rows.append('| '+method+' | '+' | '.join(cells)+' |')
        return '\n'.join(rows)+'\n'

    intro = """# 全数据集方法对照：截至恢复记录 2026-10-31

**V3 / witness 是仍保留的金融研究候选；它不是所有数据集、所有观察模型上的最优方法。**
时间上更晚的候选已试验，但没有建立稳定替代证据。当前公开 API 的默认值仍为 V1 / shared；R4 是历史任务配方。研究讲解以 [V3 方法](source-reading/METHOD_V3.md)为主。

单元格为 **AUROC / AP**，越大越好；AP 是 average precision。这里只整理已有结果，没有新运行、重算指标或据标签挑选新方法。— 表示该运行体系中未恢复到结果，不能跨观察模型补数。
历史任务已被查看，只能作为探索证据。金融是 silver 编辑标签；HalluRAG 是句标签投影；它们不能等同人类自然 token 错误标注。
这些现成标注可以直接用于评价；标签来源与历史开发暴露只是解释其证据范围，不是禁止使用。按用户指定，HalluRAG包含答案的891答为与RAGTruth对照的主组；全部及不含答案组继续单列。[R4/V3主表、错例与0.8目标](source-reading/PERFORMANCE_REVIEW.md)。

## 1. Llama：R4、V1、V2、V3 与控制

前三列为 RAGTruth 三任务完整 TEST；金融为 FRED 发布的 FinQA 162答 / TATQA 130答 silver 测试人群，不是原始 FinQA/TATQA 整个基准。原生成问题和提示未恢复，统一采用 source-only completion prompt，不能宣称复现原金融问答系统。前五列来自论文第6页表2，概率基线来自第21页表17。
HalluRAG的V3来自第29页表35的新匹配SDPA采集；R4完整TEST及分层来自本次恢复的历史报告。两者人口/token计数一致，但采集后端不同，不能当同观测配对实验。V3 span permutation 为实验账本的固定段内置换消融，保留六位打印值。R4 使用额外 local/route 观测和无标签参考校准，预算与 full-gap 方法不同。
旧 RAG 观测生产者身份未全部恢复，不宣称全链路独立复现。

"""
    methods=['R4','Raw gap','V1 shared','V2 persistent','V3 witness','V3 c=0','Fixed clipping','V3 span permutation',
             'Matched local','Symmetric local','Fixed HMM','Partial','Source NLL','Source entropy']
    text=intro+matrix(['llama_historical','llama_hallurag','llama_hallurag_r4_historical'],TASKS,methods)
    text+="""\nV3 相对 V1 的 AUROC 增量约为：QA +0.0020、Summary +0.0368、Data2txt +0.0561、FinQA +0.0744、TATQA +0.0742（由表内四位数相减，非置信区间）。
R4 仍在 QA / Data2txt 更强；QA 上 V2 的 AUROC/AP 也高于 V3。V3 比简单 clipping 的金融增益配对区间跨零，未建立稳定优越性。
金融 V3 AUROC 的历史95%区间约为 FinQA [0.7690, 0.8350]、TATQA [0.7927, 0.8919]，都跨0.8。

## 2. Qwen：密集执行主比较（六组）

论文第7页表3及第26页表26；保留原三位打印精度。V1/V2/R4 在这套观测上没有恢复到对应结果，不填入旧 Llama 数值。
七种 source-gap 读出共用与 LUMINA 完全相同的 source-present 前向，并使用匹配 source-absent 前向。
LUMINA 额外使用另一条来源及内部表示；属于输入合同适配后的官方代码比较，不是原论文 checkpoint 指标复现，也非相同观测预算。

"""
    text+=matrix(['qwen_dense'],TASKS)
    text+="""\n这套金融上 Matched local 的点值略高于 V3，但四个 AUROC/AP 配对95%区间均跨0。
HalluRAG 上 Source NLL 的 AUROC 高于 V3：差 +0.0439，配对区间 [+0.0043,+0.0866]；AP 差的区间跨0。
LUMINA 的 HalluRAG 点值也高于 V3。不能从旧金融结果推出 V3 跨观察模型全面最优。

## 3. V3 之后的候选（旧 Llama 五组）

Token state、Adaptive token、Onset 的完整五组数值从私人历史中的实际工具输出恢复；发布的证据摘录只含数值行和文件哈希，不发布私人会话。
Responsibility 与 Conditional 来自实验账本。— 不是零，也不表示已失败；Conditional 没有触发后三组自然 RAG 采集。

"""
    text+=matrix(['llama_later_candidates'],TASKS[:5],
                 ['V3 witness','Responsibility','Conditional','Onset','Token state','Adaptive token'])
    text+="""\nResponsibility、Token state、Adaptive token 在五任务的两个点指标均降低；Onset 的 QA AUROC 上升但 AP 下降，金融及另两 RAG 任务未通过原门槛。Conditional 两金融指标下降。均未晋升。
proof_revision 在2992答资格检查中找到0条可用 incompatible 证明，没有检测 AUROC/AP；assertion parser 的6773文本资格检查也未通过。代码存在不等于实验有效。
ClaimDB 只进入解析资格检查；HaluEval / RAGognize 只准备了数据，RAGHallu 身份未定；不将这些写成已完成检测的数据集。
更早的关系约束、层级等金融探索与完整历史见实验账本；本页覆盖主版本、后续主要替代候选及论文现存控制，不宣称所有历史临时实验都拥有全数据集结果。

特别保留会影响下一轮设计的旧关系专家失败结果（2026-10-28 v4）：同片段共享关系世界并积分，完整292答。与本次提出的关系观测有实质重叠，不能重新命名为已发现的新信息。下列对照来自同次匹配SDPA采集，不能用历史V1值替换其控制。

"""
    text+=matrix(['llama_relational_archive'],['FinQA','TATQA'])
    text+="""
当时大量关系null对正常与错误token都接近原来源，分离反而恶化。新方案最多是资格收紧和融合方式改变的再检验；必须与旧world_shared在同一新观测上比较，不能跳过这个失败先例。

"""
    text+="""

## 4. HalluRAG：两个元数据分层

来自第29页表35。Available / Unavailable 各891答；是发布的检索可用性元数据，不是新的语义审计。不能把分层当两份独立复现。

"""
    text+=matrix(['llama_hallurag','llama_hallurag_r4_historical'],['HalluRAG','HalluRAG_available','HalluRAG_unavailable'])
    text+="""\n恢复的R4完整TEST为全部0.576780/0.262574、包含答案0.807066/0.518830、不含答案0.401882/0.199319。此前账本0.472617/0.287484属于VAL DEV132答/6文章，不能替代这组完整TEST。
Qwen dense V3的第27页表28分层：包含答案0.7134/0.3773，不含答案0.4072/0.1531，各891答；不是cached存档。
RAGTruth没有当前适配器统一读取的等价answerability标志，本次保留原split/task，不能把“与Hallu含答案对照”写成RAGTruth每条已单独审查证据充分。
"""
    text+="""\n## 5. 分辨率删除诊断：不能晋升为新有效方法

密集 Qwen 第30页表38。Original 是原分数；Span mean 抹掉片段内差异；Span centered 去掉片段均值。
V3 均值 AP 在六组中的五组上升，但金融 AUROC 下降；Data2txt AP 的实际微小差为 -0.0000087，四位打印看起来相同。
均值的同片段 AUROC 恒为0.5；centering 保持局部排序却降低全局指标。因此不能把 pooled AP 上升解释为局部识别改善。

"""
    text+=matrix(['qwen_dense_deletions'],TASKS)
    text+="\n历史 Llama 均值删除（第29页表37）：\n\n"
    text+=matrix(['llama_historical_deletions'],TASKS[:5])
    text+="""\n## 6. Qwen cached 存档：不与密集 LUMINA 混比

第32页表41。与密集运行使用同一观察模型、原始人群，是执行变体，不是第三个独立模型复现。
cached 适配器未通过原定 dense-equivalence 门槛；保留其内部匹配结果，主比较使用上面的密集执行。

"""
    text+=matrix(['qwen_cached_archive'],TASKS)
    text+="""\n## 覆盖与可追溯性

| 人群 | 回答数 | 来源组数 | Llama 有效token | Qwen 有效token | 标签 |
|---|---:|---:|---:|---:|---|
| QA | 900 | 150 | 124817 | 126075 | 人工跨度 |
| Summary | 900 | 150 | 121248 | 123653 | 人工跨度 |
| Data2txt | 900 | 150 | 178343 | 180474 | 人工跨度 |
| FinQA | 162 | 142 | 16047 | 19425 | silver 编辑 |
| TATQA | 130 | 110 | 7747 | 9389 | silver 编辑 |
| HalluRAG | 1782 | 81 | 69483 | 71210 | 自动句标签投影 |

HalluRAG Llama 已知62649、unknown6834；Qwen 已知64269、unknown6941。unknown 参与推断/覆盖，不进入有标签指标。来源组依据已恢复的文本身份，不一定等于独立公司/报告。有效token随 tokenizer 不同；不能跨模型混合平均。

可下载 [CSV](results/comparison.csv)、[JSON](results/comparison.json)、[逐行证据摘录](results/evidence.json)。JSON 每行保留观察模型、执行组、标签类型、来源表号/历史记录位置和打印精度。
原33页研究稿保存在用户桌面01_Paper；原会话仅在04_History。本页是恢复汇总，没有重新验证原始分数/标签。
生成命令：python tools/render_method_comparison.py。
"""
    (ROOT/'docs/RESULTS_CURRENT.md').write_text(text,encoding='utf-8')
    print(json.dumps({'records':len(records),'groups':sorted({r['group'] for r in records}),
                      'new_experiments':False},indent=2))


if __name__ == '__main__':
    main()
