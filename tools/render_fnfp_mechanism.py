"""Render post-evaluation R4/V3 diagnostics; never score, fit or select a detector."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'docs/results/fnfp_mechanism_summary.json'
DOCUMENT = ROOT / 'docs/source-reading/FNFP_MECHANISM.md'
TASKS = ['QA', 'Summary', 'Data2txt', 'FinQA', 'TATQA']


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '|' + '|'.join(['---'] * len(headers)) + '|',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def main():
    evidence = json.loads(EVIDENCE.read_text(encoding='utf-8'))
    tasks = evidence['tasks']
    assert list(tasks) == TASKS
    assert all(tasks[name]['population']['unknown_tokens'] == 0 for name in TASKS)
    assert sum(tasks[name]['population']['responses'] for name in TASKS) == 2992
    assert sum(tasks[name]['population']['tokens'] for name in TASKS) == 448202
    digest = hashlib.sha256(EVIDENCE.read_bytes()).hexdigest()

    counts = []
    budgets = []
    common = []
    active = []
    transitions = []
    spans = []
    for name in TASKS:
        row = tasks[name]
        budget = row['fixed_alarm_budget']
        a = row['active_input_diagnostics']
        methods = row['metrics_diagnostic_only']
        format_counts = lambda method: '/'.join(str(method['top5pct'][k]) for k in ['tp', 'fp', 'fn'])
        counts.append([name, row['population']['responses'], row['population']['tokens'],
                       row['population']['positive_tokens'], budget['ceil_tokens'],
                       format_counts(methods['r4']), format_counts(methods['v3'])])
        budgets.append([name, budget['unavoidable_minimum_fn'],
                        f"{100 * budget['maximum_token_recall_at_budget']:.2f}%"])
        both = row['positive_overlap']['both_FN']
        primitive = row['common_FN_primitives']
        nonpositive = primitive['both_raw_gaps_nonpositive']
        common.append([name, both, primitive['raw_top5_quadrants']['neither_raw_top5'],
                       nonpositive, f'{100 * nonpositive / both:.1f}%'])
        active.append([name, a['r4_actual_input_view'], a['r4_FN_active_point_top5'],
                       a['r4_FN_available_raw_top5_but_active_point_below_top5'],
                       a['v3_FN_active_full_top5'], a['v3_FN_local_only_top5_not_an_input']])
        r4_transition = a['r4_active_point_to_final']
        v3_transition = row['v3_alternative_readout_contrasts']['raw_full_to_v3']
        transitions.append([name,
            f"{r4_transition['positive_lost']} / {r4_transition['positive_recovered']}",
            f"{v3_transition['positive_lost']} / {v3_transition['positive_recovered']}",
            r4_transition['negative_promoted'], v3_transition['negative_promoted']])
        hits = row['span_hit_diagnostics']
        format_hits = lambda method: (f"{method['any_hit']}（{100 * method['any_hit_rate']:.1f}%） / "
                                      f"{method['complete_hit']}（{100 * method['complete_hit_rate']:.1f}%）")
        spans.append([name, hits['r4']['observable_spans'], format_hits(hits['r4']), format_hits(hits['v3'])])

    route = []
    for name in ['QA', 'FinQA', 'TATQA']:
        row = tasks[name]
        stats = row['r4_stage_transitions']['r4_source_to_r4_base']
        metrics = row['metrics_diagnostic_only']
        route.append([name, metrics['r4_source']['top5pct']['tp'],
                      metrics['r4_base']['top5pct']['tp'], stats['positive_lost'],
                      stats['positive_recovered'], stats['negative_promoted']])

    examples = []
    for example in evidence['source_grounded_examples']:
        ranks = example['global_ranks']
        examples.append([example['task'] + ' ' + example['sample_id'],
            example['source_fact'], example['response_fact'],
            example['token'].strip(),
            f"{ranks['raw_full']} / {ranks['raw_local']} / {ranks['r4_point']}",
            f"{ranks['r4']} / {ranks['v3']}", example['task_top5_budget']])

    text = '''# R4与V3为什么漏报、为什么误报

当前证据指向几类不同问题：有些错误关系没有表现为强原始对比；有些局部视角根本没有进入检测器；还有些已有信号在片段聚合、route组合或上下文推断之后改变了全局名次。与此同时，这些操作也救回了大量原始对比未排到前面的正token。需要按输入、读出、关系归属和评价粒度分别解释。

本页只分析**五项完整TEST**：RAGTruth三任务各900答，FRED FinQA162答、TATQA130答，共2992答、448202个有标签token。HalluRAG的V3完整缓存已恢复，但仍缺少R4原始预测；V3单独审计见[HalluRAG FN/FP](HALLURAG_FNFP.md)，不混入这里的R4/V3配对机制计数。所有分数来自已经冻结并完成评价的缓存；没有新观测、训练、权重/阈值拟合或新方法选择。已暴露人口继续是探索证据。逐例来源核对见[FN/FP案例](FNFP_CASES.md)，机读统计见[诊断JSON](../results/fnfp_mechanism_summary.json)。

## 如何定义这次FN和FP

每个任务、每个被检查的通道或阶段，分别按分数降序取 `ceil(0.05*N)` 个告警，同分保持原token顺序。五组均无unknown，因此原评分人口与此次有标签评价人口一致。TP/FP/FN按已发布的正字符跨度映射到原token；RAGTruth为人工跨度，FRED为合成编辑跨度，两类现成标注都可用于评价，含义分别保留。预算属于完整任务，不能在一个案例内重新排序，也不沿用旧48答DEV的top10%计数。

下表单元格为**TP / FP / FN**。FP表示未与发布正跨度相交，不自动等于独立人工确认的正确事实；FN也可能是一个错误引用/关系跨度中本身受来源支持的词。

'''
    text += table(['任务', '回答', 'token', '正token', '告警预算', 'R4 TP/FP/FN', 'V3 TP/FP/FN'], counts)
    text += '''

预算本身给出了不可避免的漏报下界：正token数为P、预算为K时，最少FN为 `max(0,P-K)`，最高token召回为 `min(1,K/P)`。这只是理想排序的容量界。它不使现有误报合理，也不能解释AUROC未达标；AUROC没有这个固定阈值限制。

'''
    text += table(['任务', '理想排序仍至少FN', '预算允许的最高token召回'], budgets)
    text += '''

## 原始信息不足与关系归属错误

下面统计两方法共同漏掉的正token。`full≤0且local≤0`表示在各自历史对比条件下，来源存在没有降低这些已生成词的预测概率；它不证明回答受来源支持，也不证明该词本身是假事实。需要区分语义关系判断和词的可预测性。

'''
    text += table(['任务', '共同FN', '两个原始视角均未入top5%', 'full≤0且local≤0', '占共同FN'], common)
    text += '''

引用错配、实体—事件绑定、时间范围、条件或作用域错误，可以由一串来源中确实出现过的词组成。一个广泛标正的错误跨度也会覆盖这些词。例如QA12297的 `able` 所在文字在来源Passage3中存在，问题涉及回答对段落的归属；不能把其负gap解释成“模型没有识别一个本身错误的单词”。同样，混有结构化字段和用户评论的Data2txt来源可能含有互相冲突或不同时间的陈述，不能只因词出现在一条评论里就宣称整句成立。

上表是**原始对比与发布标签之间的诊断**，不是对所有共同FN重新完成了语义裁决；这些数量不能作为某一种语义错误的总体发生率。

## 输入视角缺失与读出后的名次下降

R4的QA、Data2txt及两金融任务先取local/full平均形成实际输入point；Summary只使用full。V3只使用full。分别对原始视角做top5%排序，与对实际输入或最终分数排序，是不同的排序问题。

'''
    text += table(['任务', 'R4实际输入', 'R4实际point已top5%仍FN',
                   '某原始视角top5%但point未入', 'V3 full已top5%仍FN', 'V3 local-only top5%未使用'], active)
    text += '''

Summary有129个R4 FN仅在local视角进入top5%，其full没有进入；这是现行full-only配方未使用的观测，不能说成R4已经接收后又压掉。其余pair配方也必须先检查平均后的实际point，不能把“任一单独视角很高”直接算作后端抑制。

实际输入到最终读出之间，也同时有正token的进入和退出。下表“丢失/救回”是两个固定预算告警集合的差集；“新增FP”指原始实际输入未告警、最终告警的负token。这不是干预实验，更不把各阶段当成相互独立的原因。

'''
    text += table(['任务', 'R4正token 丢失/救回', 'V3正token 丢失/救回',
                   'R4从point未告警新增FP', 'V3从full未告警新增FP'], transitions)
    text += '''

因此，局部“强信号掉出预算”是真实存在的，但它不能推出整段共享或上下文推断整体无益。V3的cached score可以写成现有 `witness_emissions(full)` 加上差值项，用来追踪自身证据与上下文分数；差值不是独立观测，也不是已经实施了删除上下文的反事实实验。V1/shared、V2/persistent、V3/witness是不同读出，不能把它们之间的比较写成连续执行的三个阶段。

## R4金融迁移：route组合处发生了什么

当前金融R4沿用QA冻结参考数组和QA配方，`base = 0.75*source + 0.25*calibrated_route`。参考数组的哈希已与历史score plan核对。source是原始片段prior经无标签参考CDF变换的量；它不是错误概率。下面比较同一缓存中source与base的任务告警集合。

'''
    text += table(['任务', 'source阶段TP', 'base阶段TP', '正token退出', '正token进入', '负token进入'], route)
    text += '''

同一组合在QA增加了预算内正token，在两金融任务却明显减少。Summary/Data2txt的route权重原本为0，source与base相同。可以确认**这一步的排序变化与金融退化相伴**；原因可能涉及route信号所表达的量、参考分布迁移或金融表格/推导文本与自然QA的差别，需要分别检查。目前没有按这些标签拟合新权重，也没有把source阶段晋升为新检测器。

R4其他阶段也有双向变化：point到片段prior、base到带窗口创新的unary、unary到最终连续性分数都会改变全局名次。完整的退出/进入计数保留在JSON中。不能仅选择一个最终漏报案例，把某个阶段所有作用都解释成有害平滑。

## 具体错误关系怎样落到分数上

以下来自当前TEST，已对照原来源和回答。排名1表示任务内最高风险，是否告警按前述完整任务预算判断。

'''
    text += table(['任务/ID', '来源', '回答', '定位token', 'full/local/R4 point排名', 'R4/V3最终排名', '预算'], examples)
    for example in evidence['source_grounded_examples']:
        text += ('\n\n**' + example['task'] + ' ' + example['sample_id'] + '来源短引**：\n\n'
                 + '\n\n'.join('> ' + quote['text'] for quote in example['source_quotes']))
    text += '''

QA15716把公元前的时间改成“几百年前”；两个原始视角及R4实际point排名1492都在预算6241内，片段prior排名却变成116867，最终为114453；V3自身发射量约2.6324、上下文差值约−2.2830，最终排名20219。这里确有实际输入到后续读出的名次下降。

Summary410把第一台机器人的路径进度归给第二台；local排名844在预算6063内，而full/R4实际point排名81947本来就不在。当前R4 Summary和V3都只读full，因此这个token首先暴露视角未被使用及实体—事件绑定问题，不能说local信号被R4后端压掉。

Data2txt8937的来源显式不接受预约，回答却说接受。full单独排名7962在预算8918内，但R4实际pair-point排名12232**在prior之前已不在预算内**；prior排名85483、最终87425仍低，不能把最初掉出预算归因于prior。V3确实读取full：自身发射量约0.9702，加上−2.0498的上下文差值后最终排名101511。两个方法的失败链不同。数值链说明名次怎样变化，来源核对说明错误是什么；两者不可相互代替。

## 捕获一个词与捕获完整错误短语

对每个发布正字符跨度，任意相交的已评分token中命中至少一个记为“任意命中”；所有这些token均告警才记为“完整命中”。不可观察跨度另列，五组此次均为0。下表不替代官方token指标，重叠跨度也不能视为独立样本。

'''
    text += table(['任务', '可观察正跨度', 'R4任意/完整命中', 'V3任意/完整命中'], spans)
    text += '''

V3在三项RAG任务触及更多不同错误跨度；这不必伴随更多正token命中或更高AUROC。比如长错误段中的很多token被同一次告警集中覆盖，与更多短错误段各找到一处，是两种定位行为。金融中的错误账户名、年份或频率也可能只找到首子词；“整个错误没发现”和“错误短语只发现一部分”应分开报告。FP同样可能落在正确推导、来源支持的实体词，或一段真假混合陈述的正确部分，需逐例查证。

## 下一步模块必须先通过哪些检查

这些是下一轮需冻结的验证条件，尚未实现、采集或据最终标签挑选模块。允许模块组合和任务专用配置；继续坚持无监督，冻结模型、提示、输入单元、参数、评分人口及完整分数后才读取新的最终真值。当前五组继续作探索证据，不能重新命名为未接触确认集。旧RAG原始观测生产者身份仍有历史缺口，本轮也没有重做后端匹配的LM观测；可复核缓存运算与阶段哈希，不等于已经消除了所有观测混淆。

| 待检验问题 | 冻结的模块/观测合同 | 必须比较和报告 |
|---|---|---|
| 词都可预测，却属于错误引用、实体或时间 | 核验声明及其来源归属；固定基础模型、提示、句子/声明边界，输出来源引用、归属关系、支持/冲突/unknown，先覆盖所有单元 | 核验器单独与原R4/V3分开评价；逐例核对引用、实体—事件、时间与条件；引用存在不等于蕴含，模型意见不当gold |
| 某个局部观测可用，实际读出没有读取 | 固定local/full的含义、后端、原token轴与成本；先确认差异是否来自前缀条件或执行方式 | 分开报告未输入的视角、平均形成point的变化和point之后的变化；原分数与预算同时保留，不按最终标签扫权重 |
| 个体信号与片段/上下文存在冲突 | 保留同一声明内每个token的自身证据和区域证据；先检查已知计算身份与可复现缓存 | 对强信号FN、被抬高的FP同时报告，并保留总体进入/退出；任何新读出须单独冻结完整规则，不能由单例修复宣称改进 |
| route在金融中改变了错误排序 | 记录route定义、无标签参考源、模型/后端/资源哈希和文本结构；对来源、单位、推导、表格绑定分别作固定检查 | 先解释为何同一操作在QA与金融不同；若新增模块，单列成本和控制；本轮不从已暴露标签拟合经验权重 |
| 固定预算与跨度长度掩盖定位差异 | 保留主AUROC/AP、固定top5% token预算，同时声明跨度任意/完整命中等次指标 | 给出预算理论界和不可观察范围；不通过改预算提高AUROC，也不把误报压到一个案例内比较 |

核验器或新观测若没有提供超出原信号的可复现信息，就不能把组合本身当成贡献。当前数值链与案例核对只回答失败发生在哪里、哪些解释仍待验证；没有建立新的效能结论。

'''
    text += (f'证据JSON SHA256：`{digest}`。生成命令：`python tools/render_fnfp_mechanism.py`。'
             '\n原始诊断命令、完整缓存绑定及120条按规则选取的案例保存在本地 '
             '`outputs/fnfp-mechanism-20261102/`；公开JSON保留聚合计数与来源收据哈希，未复制私人历史。\n')
    DOCUMENT.write_text(text, encoding='utf-8')
    print(json.dumps({'document': DOCUMENT.relative_to(ROOT).as_posix(),
                      'evidence_sha256': digest, 'tasks': len(tasks),
                      'new_detector': False}, ensure_ascii=True))


if __name__ == '__main__':
    main()
