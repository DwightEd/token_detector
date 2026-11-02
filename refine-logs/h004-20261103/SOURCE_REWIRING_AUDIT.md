# 来源值重接的生成似然对照：历史重叠与准入审计

2026-11-03，Asia/Shanghai。只读审计；先仅输出既有 31 条 prior ID，再检查 P006/P007/P008/P021/P024/P025/P026 记录及有关实现。没有输出完整历史索引，没有打开 outputs、原始 gold、模型文件或运行新观测；仅新增本说明。H005 继续作为当前主线。

**结论：朴素“同类型值轮换 + 原回答 teacher-forced 似然差”与已拒绝的 P021 直接重叠，目前不应实现或晋升。** 换成 typed JSON、把旧 mixture 改成 max、减少 K 或更换模型，本身都不足以重新打开该假设。只有新的、可独立核验的语义兼容干预合同，才可能满足历史重开条件。成本较低不能替代这个条件，也不是性能证据。

## 已有观测到底是什么

| 历史条目/实现 | 与候选的关系 |
|---|---|
| **P021_RELATIONAL_NULL_SOURCE_WORLDS**；[relational.py](../../experiments/research/relational.py)、[relational_transfer.py](../../experiments/relational_transfer.py) | `relational_sources()` 已对表格数值执行确定性行/列轮换，保留非数值单元、标签和年份表头，百分比及币种不混合，并去掉重复/no-op 世界。`collect()` 已在同一冻结模型及 matched SDPA 下，对原始来源、空来源和轮换来源逐 token teacher-force 同一原回答，保存逐世界 logp。它不是仅仅做过 attention 或 judge 标签实验。 |
| P021 读出 | `multiworld_scores()` 把原始与扰动世界形成的观测送入 latent-world/shared 推断。拟议的 `max(logp(S')−logp(S))` 是另一读出；若世界、序列、模型和后端相同，它可由旧原始 logp 直接计算，不能称新观测。旧记录说有关存档不在本地；缺缓存不构成重跑授权。 |
| **P006_DIRECT_SOURCE_WRITE / P007_SOURCE_HISTORY_COMPETITION**；[source_write_observe.py](../../experiments/source_write_observe.py) | 前者测源写入/历史写入的激活消融，后者是既有 direct_gap 与 history_gap 的差。来源内容重接与这些激活读出在操作上不同，但“来源贡献强”仍不等于“关系兼容”，不能把换个差分命名为新语义。 |
| **P008 / P024 / P025**；[binding_transfer.py](../../experiments/binding_transfer.py) | 曾使用来源有限事实/证书、依赖边及关系因子；局部门槛与自然覆盖问题已知。候选不是这些证书的同一实现，但“类型保留→绑定正确”的跃迁会重犯同一问题。P024 的已成功局部控制也不能再当新发现重算。 |
| **P026_RETROSPECTIVE_REPAIR_AND_PREFIX_RESET** | 旧方法改回答候选/前缀，再做来源有无的未来兼容比较；候选固定完整原回答、只改来源，因此不等同于该 response-repair 操作。这个差别并未避开更直接的 P021 重叠。 |

P021 的记录已明确拒绝“再 shuffle numeric cells”及“把 null-world mixture 改名关系核验”。其重开条件是**新的干预，具有独立冻结的语义兼容合同、更广的关系覆盖和 no-op 对照**；仅词袋保留不够。本审计不复述旧 AUROC 数字，也不重新计算任何结果。

## 候选可定义，但没有无条件的事实识别保证

设 `R(S)` 是固定的、可逆且保留类型/路径的规范来源表达。保持原回答 token 轴 `y_1:T`、原始生成上下文 `Q` 和完整 teacher-forced 前缀不变，定义

\[
\ell_t^{(k)}=\log p_\theta(y_t\mid Q,R(S_k),y_{<t}),\quad
d_t=\max_{k=1..K}\{\ell_t^{(k)}-\ell_t^{(0)}\},\quad S_0=S.
\]

正值只表示某个替代来源让模型更容易预测这个已给定 token。它不是“原来源不支持该 token”的逻辑判定，也不是事实错误概率；无变化不能解释为正确。K 必须由输入来源及冻结规则决定，不能按回答、错误位置、任务 gold 或试跑效果选最有利的交换。

原始 `S` 与 `R(S)` 的序列化变化会改变模型行为，所以 **`R(S)` 下的原世界是必要 matched-format 基线**；不能拿旧 raw-source R4 观察直接充当其分母。这个必要基线是新输入的工程对照，不是新机制的证据。规范化与重接两个作用必须分开。叶值多重集保留也不自动等于 tokenizer 的 BPE 词袋完全相同，后者需要逐字节/逐 token 实测，不能口头担保。

**一个有条件、可推翻的主张：** 在来源 schema 提供明确实体—属性—时间角色、交换不会破坏该 schema 的解释、且模型能利用这些绑定时，把一个错误复制值重接到它在回答中声称的角色，应提高相应回答值的条件似然；相反方向的正确绑定应降低。这个主张只关乎冻结模型对来源干预的响应。使用辅助模型时，它不识别原生成器的致错原因；即便同一模型，也不自动证明其原始采样路径的原因。

可区分的构造是来源 `A→10, B→20` 与 `A→20, B→10`，固定问题询问 A，同时交叉检查回答 10 和回答 20 的似然方向。两来源含相同值，期望的交换交互来自角色敏感性而非数值是否出现。源自 schema 的两个世界及两种答案都必须在观察前完整冻结，不能根据模型分数选交换。它只是可能的资格合同示例，当前不建议启动；是否与 P021/P024 既有材料重合还需逐指纹核查。没有这个新合同，按 JSON leaf kind 分组轮换只是已测方法的语法变体。

## 不可识别性和必须承认的失败

- **同类型不保证合法关系。** `year` 与 `amount` 都可能是整数；`weekday` 与 `opening_time` 都可能是字符串；不同币种、百分数/计数、开始/结束时间、总额/分项也会被错误交换。类型树可逆只证明转换保真，不证明替代来源语义合法或回答真值发生了预期变化。旧 P021 已比纯 leaf-kind 更细地保留表头、币种及百分比，仍未解决语义兼容。
- **来源不敏感。** 若所有世界下 `p_theta(y_t|...)` 相同，则 d_t=0，正确、错误以及模型根本没读懂来源完全不可区分。统一错误且高置信的模型也会如此；不能把零当“已核实”。
- **错误地响应也会有大差值。** 模型可能偏爱某个位置、数值大小、熟悉措辞或与预训练知识一致的世界，使正确 token 在一个不相容世界中似然反而更大。值袋相同不能排除这些解释。
- **teacher-forced 历史仍然在。** 固定 `y_<t` 可以保住原 token 身份，但早先错误关系也可能支撑后续连贯词；重接不保证在后续词上显出差异，不能再偷偷加入 prefix reset 冒充修复。
- **max 受机会数影响。** 增加 K 会使 max 不减，即使没有增加事实信息。不同来源的有效交换数、重复值、no-op 或不合法世界必须全量列出；不能把无可用交换的案例赋零风险并称正常。没有统一 fallback 及覆盖合同前，只能报告诊断，不能形成全 token 检测器。
- **只有类型化来源可操作。** H004 严格完整来源语法目前只有 3/34 可转换。嵌入式表格和普通文本不能默认为容易重接；将其排除会大幅改变评价人口。增加自由抽取器又是一个新主要模块，不能在此轮默默加入。

若将来独立合同真的成立，最小判别应同时保留：原世界/规范化原世界、schema 合法的反向绑定世界、同值或非目标字段的 no-op 控制，以及正确端与错误端。它们用于排除格式/位置/词汇与关系响应的混淆；不能在已看 benchmark 标签上选 K、schema 分组或 max/mean 赢家。完整 token 和 unknown/fallback 规则必须另行冻结，绝不经验加权 R4/V3。

## 一手文献：相近操作与不能借来的结论

| 论文 | 建立的内容及边界 |
|---|---|
| [**ContextCite: Attributing Model Generation to Context**](https://papers.nips.cc/paper_files/paper/2024/hash/adbea136219b64db96a9941e4249a857-Abstract-Conference.html)（NeurIPS 2024） | 通过上下文子集消融与响应似然建立贡献归因；论文 Eq.1 用原回应的 log-probability drop 评价归因。它明确允许找到被模型误解的来源。贡献归因不等于来源蕴含，不能据此宣称重接差值是事实真值。值绑定重接也不同于删除上下文子集。 |
| [**Trusting Your Evidence: Hallucinate Less with Context-aware Decoding**](https://aclanthology.org/2024.naacl-short.69/)（NAACL 2024，CAD） | 对比有/无上下文输出分布以改善生成，是候选及既有 R4 source-gap 的直接概率层近邻。它不是原始固定 token 的可靠关系错误检测定理。用替代绑定替换空来源会改变对照，却仍属于来源条件分布对比。 |
| [**Pointwise Mutual Information Based Metric and Decoding Strategy for Faithful Generation in Document Grounded Dialogs**](https://aclanthology.org/2023.emnlp-main.639/)（EMNLP 2023） | 研究文档对回应条件概率的 PMI 贡献及相应解码。历史 PMI/source-gap 已做过的内容不能靠换名复验；在文档贡献强时照样可能复制错误角色。 |
| [**Understanding and Detecting Hallucinations in Neural Machine Translation via Model Introspection**](https://aclanthology.org/2023.tacl-1.32/)（TACL 2023） | 使用来源扰动产生对照并研究来源贡献的内部症状，之后检验自然幻觉。其问题包含大幅脱离来源的翻译，不证明同词袋局部账号/年份绑定重接有效。模型对扰动的响应与语义错误仍需分开。 |
| [**Speculative Decoding with Big Little Decoder**](https://proceedings.neurips.cc/paper_files/paper/2023/hash/7b97adeafa1c51cf65263459ca9d0d7c-Abstract-Conference.html)（NeurIPS 2023，BiLD） | 小/大模型通过 fallback/rollback 协作降低生成延迟。它不验证来源事实，也不为四次 teacher-forced 来源遍历提供准确性保证。另一个同名 [**BiLD: Bi-directional Logits Difference Loss for Large Language Model Distillation**](https://arxiv.org/abs/2406.13555) 是蒸馏损失，同样不是来源重接核验；两者不能混引。 |
| [**Monitoring Latent World States in Language Models with Propositional Probes**](https://openreview.net/pdf?id=0yvZm2AjUr)（ICLR 2025） | 组合地读取概念及绑定，表明模型输出错误时内部仍可能保留忠于上下文的命题。它使用训练出的 probe 和有限谓词闭世界，不是无训练 leaf 轮换检测器，不能直接搬进 ground truth 仅评价的本项目。 |
| [**Legible Failures: Detecting and Repairing In-Context Binding Errors**](https://arxiv.org/abs/2609.11216)（2026 预印本） | 在特定实体—义务任务中用训练探针、验证折选层和 query-entity 反事实区分 token 出现与绑定读取。它支持“绑定与输出 confidence 可能分离”的动机；其训练/选层协议不满足本项目无监督约束，不能据其结果推导本候选有效。 |

上述文献没有提供本候选的新颖性证明。最近的约束首先来自本仓库已经完成的 P021，而非文献是否使用完全相同的命名。

## 计算预算与决定

若 K=3，则原规范世界加三个替代世界是 **四条整答 teacher-forced scoring 路径**；这不等于四个物理 GPU forward，prefill、分块 continuation、上下文长度及内存实现仍决定实际调用数。它有机会比对约 22k 焦点单元各做六标签排列便宜；这只是待测工程成本判断，不是准确性或 AUROC 承诺。若另加 raw-format 基线、空来源或完整控制，成本还要单独增加；完全相同观察必须复用。

**当前决定：不实现、不注册模型运行、不替换 H005。** 纯类型轮换不满足 P021 的重开条件。可保留的研究问题是“如何在不读取目标标签的前提下，提供独立可信的来源角色/兼容世界合同，并区分模型绑定失败与格式敏感性”；这仍未解决，也可能需要比四次似然遍历更多的证据。任何未来准入先交代与旧记录的实质不同，再冻结方法、人口、阈值、参数与分数；gold 只在封存后的评价中出现。
