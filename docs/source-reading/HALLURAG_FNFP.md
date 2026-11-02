# HalluRAG：实际 FN / FP 为什么出现

[五任务机制诊断](FNFP_MECHANISM.html) · [逐例事实核验](FNFP_CASES.html) · [本页统计与来源证据 JSON](../results/hallurag_fnfp_evidence.json)

这里分析已冻结的 V3 witness 与原始 full_gap 缓存。主要组按发布的 answerable / target-available 元数据确定：**891 个回答、28,844 个已知标签 token，其中 6,311 个正 token**。全部 1,782 个回答的 69,483 个分数均已恢复；主要组 994 个 unknown 只在评价时排除。R4 原始预测缓存被历史清理删除，[历史聚合结果](../results/hallurag_strata_evidence.json)仍保留，本页没有新恢复的 R4 案例。

## 先看告警预算与标签范围

global top 5% 使用 `ceil(0.05 × 28844) = 1443` 个告警，同分按原始回答和 token 顺序稳定处理。

| 冻结分数 | TP | FP | FN | token precision | token recall |
|---|---:|---:|---:|---:|---:|
| V3 witness | 752 | 691 | 5,559 | 52.11% | 11.92% |
| 原始 full_gap | 486 | 957 | 5,825 | 33.68% | 7.70% |

即使排序完美，1,443 个告警最多覆盖 6,311 个正 token 中的 **22.86%**，至少留下 **4,868 个 FN**。V3 的 5,559 个 FN 比这个预算下限多 691 个。大量 FN 的数量不能直接等同于大量独立事实被漏掉。

原生标注是**句级判断**：现有 `sentence_token_labels` 将正句广播到每个相交 token。名字片段、地点、功能词和标点都可能继承正标签；资料没有最小错误字符跨度。

## FP：正常开头已获得极高的原始分数

691 个 FP 中，**204 个在首 token，396 个在前五个 token**。按 final 从高到低、每个新 source 取第一例得到的五个极端 FP，全是开头的 `" I"`，来自同一个 Llama-2-7b-chat-hf int4 生成器。下面保留两个有来源证据的例子。

- **FP3，source 475：** 来源列出 OWCS 的奖金池、路线图和观众保持问题；回答也复述了这三点。误报却落在前置句 “I'm just an AI…” 的首 token，该原生句标为 `non-hallucinated`。原始 `d=18.6870` 已极高，context 仅增加 `0.6201`，final 为 `18.6140`。观察到的信号主要是开头措辞对上下文的预测变化。
- **FP4，source 520：** 来源明确写道 **“Two Night Stands” 将于 April 1, 2024 作为 Morgan 的第二首乡村电台单曲发布**。回答后文却说来源未提及第二首单曲，相关句标为 `hallucinated`；当前 FP 选中的另一句是正常的 “I'm happy to help!”。首 token 的 `d=18.0623`、context `+1.2213`、final `18.5905`。这个案例展示了 token 告警位置与真正错误命题的偏移。

这五例提供了 source 多样性，尚未提供生成器多样性。它们不能代表所有生成器的极端误报模式。

## FN：来源支持的词落在含错误命题的句子里

按 final 从低到高、每个新 source 取第一例得到五个极端 FN。以下两例可以直接对照来源中的事实。

- **FN3，source 518：** 来源写明 “On 1 April, Pakistani police said that they had arrested **12 people** …”。回答却说没有关于警方四月一日行动的信息，形成来源可直接反驳的否定断言。所选 FN token `" Hey"` 来自句子前半部的标题 **Richard Heyn**，这个名字确实在另一段来源中。它的 `d=-8.1153`、context `-2.9846`、final `-3.6775`。句级正标签把后面错误的否定断言同时传播到了前面有来源支持的名字词。
- **FN5，source 481：** Portland 来源只说 **one man** 开车冲向示威者、喷辣椒水后被拘留；另一段才出现足球人物 Nauman。回答把 **Mohammad Nauman Khan** 绑定到这起拘留事件，来源没有支持这项跨段人物—事件关系。所选 FN token `" Portland"` 本身却是正确地点：`d=-8.0553`、context `-2.3809`、final `-3.0737`。正确地点和动作的复制，并未核实人物关系。

5,559 个 FN 中有 **3,642 个 `d<0`**，表示该 token 在有来源条件下更可预测。上述案例说明，局部可预测性可以与句内否定、信息范围或实体归属错误共存。另一个极端案例的来源与发布正标签存在明显张力；审计保留原标签，没有据此重标或调参。

## 阶段量与核验

这里的 `d` 是冻结的 `log p(token|无来源) − log p(token|有来源)`；emission 使用现有 `witness_emissions(d)`，context 定义为 **冻结 final − emission**。这是代数分解，没有重跑推理或进行因果干预。

完整 witness / full_gap 的 69,483 个 token 身份、回答哈希、原始序号和字符偏移精确对应；已知标签单位中的分数与完整预测缓存逐项相等。answerability 元数据按回答 ID、source ID 和回答哈希连接，句标签投影也与原始封存结果完全一致。所有案例均为事后诊断，未改变方法、分数、预算或阈值。
