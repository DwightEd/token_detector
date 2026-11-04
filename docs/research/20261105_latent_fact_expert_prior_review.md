# 来源事实 latent experts：最近先例、反例与唯一必要切口

2026-11-05。独立只读审查；实际检索一手论文，定向阅读旧实现。未运行模型、tokenizer、方法评分或目标真值；本文不构成新候选／实验准入，不改变 H020 已关闭的固定分支。

**结论：拒绝把“Bayes 权重的来源事实混合”本身提升为新主算法。** 固定 latent source 的 posterior predictive 已是 RAG-Sequence 的链式展开；过去 prefix 的 LM likelihood 检索也已有明确先例。仍可研究的唯一新信息是：同一个原决策之前，完整来源关系的主体／动作／条件身份是否真的控制了专家后验，而非来源长度、常见值或既有回答回声。下一门应先反驳这些竞争解释，不能先大规模产一个新的 mixture 分数。

## 1. 核过的一手邻近方法

| 原始资料 | 已存在的内容 | 本议题不能冒称的新意／保留差别 |
| --- | --- | --- |
| [RAG，NeurIPS 2020，§2.1](https://proceedings.neurips.cc/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf) | RAG-Sequence 对固定 latent document 的完整序列概率边际化；RAG-Token 允许逐 token 使用不同 document。 | 固定 latent 专家的过去 prefix 后验，是前者直接可推出的预测形式，不是新理论。把专家从检索文档缩到来源事实，改变粒度与许可义务，不改变这一混合代数。 |
| [REPLUG，NAACL 2024，§3.2 Eq.2、§4](https://aclanthology.org/2024.naacl-long.463/) | 把各检索文档分别放到冻结 LM 输入前，混合各次 next-token 概率；另用 LM continuation likelihood 教 retriever。 | 多来源 frozen-LM 专家、概率级组合、额外 K 次前向均已有先例。其 retriever 权重不等于本文候选的固定 latent Bayes prefix 权重；不得谎称公式完全相同，也不能把“非等权”作为创新。 |
| [In-Context RALM，TACL 2023，§6.1 Eq.6](https://aclanthology.org/2023.tacl-1.75/) | 测试时未知未来 continuation，改用过去 prefix 的条件 likelihood 对文档作零样本 reranking。 | “只用过去 prefix，不读当前 y 就选来源”已有非常接近先例。选择一个文档与保留 posterior mixture 有区别，但这不足单独构成新机制。 |
| [CoCo，Findings EMNLP 2021](https://aclanthology.org/2021.findings-emnlp.10/) | 对 source、生成文本与语言先验建模，以反事实估计 factual consistency。 | full-source 与另一 source-conditioned 模型的 likelihood 对比不是空白。CoCo 并非本候选的 fixed-fact Bayesian mixture，因此只能认定家族重叠，不能声称逐公式同义。 |
| [Bayesian Sequential Estimation，EMNLP 2023](https://aclanthology.org/2023.emnlp-main.949/) | 用 sequential decision 决定核验 claim 还需不需要继续收集检索证据。 | “Bayesian hallucination detection”名称已存在；其 claim classifier／检索停止决策不同于原 token 的 latent-fact predictive mixture。需要区分，不能仅改名。 |

另检索到 [MoFE 的作者 v1](https://arxiv.org/abs/2110.07166v1)，是按实体／依存弧等事实性目标训练多个 summarizer experts，再作参数／logits ensemble，并非逐来源事实专家。该 arXiv 的当前 v2 已改题为 CaPE；不能用未带版本的链接仍称 MoFE。名称 “factual experts” 本身不新，这一远邻也不能用来证明当前方案已经被逐项实做。

以上只概括支持本判断的内容；不采用第三方摘要的性能或方法解读。未找到的具体 detector 不等于不存在，本文不作穷尽性新颖声明。

## 2. 标准混合能证明什么，不能证明什么

下列是自己的实数代数审查，不是论文关于幻觉真值的定理。给定 task/question x、来源许可事实邻域 F_j、同一 LM 的规范化专家分布 p_j(y_t|x,R_{<t})，以及在看 R 前固定的 π_j≥0、Σπ_j=1：

\[
 P(R_{1:T}|x)=\sum_j\pi_j\prod_{t=1}^T p_j(R_t|x,R_{<t}),\quad
 w_{j,t}=\frac{\pi_j\prod_{s<t}p_j(R_s|x,R_{<s})}{\sum_k\pi_k\prod_{s<t}p_k(R_s|x,R_{<s})},
\]
\[
 p_{mix}(y_t|x,R_{<t})=\sum_jw_{j,t}p_j(y_t|x,R_{<t}).
\]

这就是 fixed-latent mixture 的 posterior predictive。必须用原始未 warped 的规范化 LM 概率及 logsumexp；generation repetition penalty 不可偷偷当概率模型。支持不足使分母为零时必须 unknown；不能加 epsilon 再称 exact Bayes。对于固定单专家 j，整段 log-loss 有标准界 `−log P(R) ≤ −log p_j(R)−log π_j`。该界比较的是语言模型编码损失，**不保证事实真伪、token AUROC 或错误定位**。

若拟用 `g_t=log p_full(R_t)−log p_mix(R_t)`，它只是两个预测模型对已给原 token 的有符号 log likelihood ratio。不能从 proper mixture 推出“错误 g>0，正确 g≤0”。专家本身可同样错绑、完整语义前提可缺失；full-source 更能预测某个正确 token 时也会 g>0。

“posterior switch”亦非免费真值证据。更新满足 `log(w_{j,t+1}/w_{j,t})=log p_j(R_t)−log p_mix(R_t)`；正常跨事实叙述同样会大幅改变后验。更不能在看完当前 R_t 后用 w_{t+1} 再评分 R_t，冒充 prechoice 路由。未定义 source-fact transition law 而允许随意 reset／切换，就是另一个隐模型；需独立冻结，不能以“动态 Bayes”略过。

## 3. 对旧本地先例的精确差异

| 已审实现 | 已做到 | 本候选有无实质区别 |
| --- | --- | --- |
| `experiments/research/relational.py::multiworld_scores`，P021 | 来源 row/column null worlds，uniform latent world 与 Beta token responsibility；world posterior 使用完整 span evidence。已完成并拒绝，不能靠改先验／温度重演。 | 没有伪造 null world、只用来源真实完整事实且在当前 y 之前更新，信息条件有差别；但 world marginalization 不是新内容。若仅把 null 政名 fact 而不解决完整语义，仍应拒绝。 |
| `relational_shared.py::constrained_scores` | 原 token contrasts + uncertain conflict clauses 的 exact Boolean DP，返回 working-energy responsibility，非真值概率；否定等不支持语法 unknown。 | 并非上述 p_j 的原 token likelihood expert，但已覆盖“关系约束 + shared posterior”。新信息必须来自额外许可与测量，不能只换 DP 为 Bayes 包装。 |
| H005 `focused_verdict_qualification.py` | 保留原 R 的 whole／全部 exact units，核验 source 下的 fixed verdict；范围缩小结果混合，未晋升。 | 本候选切 source 并预测原 y，非同一观测；仅宣称更小 scope 仍不足。保持条件完整性与合法多事实推理是新义务。 |
| 旧 `relation_readout.py::compatibility`（N/A/G） | N 为 query 与 O-projected V cosine，A 为 head mean attention，G 为 A×V 经 O 后向 query 的投影。 | 无 latent-fact likelihood posterior，不能仅据 NAG 失败判本公式已实做；也不能借其内部 graph 给本纯文本概率混合增加并未观测的机制含义。 |

定向源码 SHA：relational `fbee3be72406683971bb5bcefab75171ec92a2cca30bf4a2ad51c0f541ef7330`；relational_shared `acaa21980df65908a92a00929e16369913a4fb16684b6a7164ecfbc013bb9f1f`；H005 collector `c58c2719c6f5f95dd1b9448f57f3e8c052e3bc4e76387086dda7944c5e69f12b`；旧 NAG `d544af3287818adf87d02eb2ce9dcdf9ade003877c70999c388de2a8fc1dc877`。此次读取既有 registry 及已知结论，不重跑其指标。

## 4. 四个会直接推翻泛化说法的反例

1. **多事实与合法算术。** S 的两条事实分别含 100、200，正确回答总计 300；每个单事实专家都缺一个 operand。若它们对 300 都低概率，凸混合也低概率。要包含许可的 joint neighborhood／推理程序，或将相关 tokens 标 unknown。随意拼邻句没有证明前提完整。
2. **跨句切换与回声。** 一段先描述 A 再描述 B，fixed-latent posterior 会将前一段积累到 B 上；错误的旧 R prefix 又能使不相干专家因词面回声取得高权。当前 token 排除出权重，仍没有排除过去错误。有限正确切换必须是反例控制，不能只测单主体 toy。
3. **full-S 专家退化。** 若加权 λ_t 的完整来源专家以保证覆盖，`p_mix(y)≥λ_t p_full(y)`，因此 `g_t≤−log λ_t`。当历史被 full-S 解释得最好，λ_t 可接近 1，方法回到 full-S 自比。删除 full-S 则重新暴露覆盖缺口；两者不能通过结果后挑选解决。全来源与单事实专家的规范化概率相容，也不等于它们的语义假设相容。
4. **重复与词频。** 对物理重复段各赋一个相同 prior，机械复制支持相同值的句子会增加其 mixture mass。必须在看 R 前给 semantic-equivalence class／重复 witnesses 的 prior 合同；若做不到就承认 multiplicity-sensitive。对 passage 位置重排不变，不能排除此偏差。

共同 prefix 在当前值前还没确定主体／谓词的 token，任何此类 past-prefix 专家法都缺少唯一 owner 身份。这不是失败后可按标签删除的 token；应预先报告 unknown。后置 owner、共指、否定范围、合法多对多与运算是前端覆盖义务，不能由高 LM likelihood 自动发证。

## 5. 唯一值得再审的必要实验（proposal-only）

**问题不是 mixture 是否能改变数字概率，而是前一刻已消费的关系身份能否支配事实 posterior。** 先只做零模型资格：从当前来源按原出处保留完整 owner／action／condition 的 fact neighborhoods，保留重复、alternative recipes 与上游限定；不能让新 LLM 生成 JSON，再把未过 H020 门的输出当许可。严格 source-only 独立审查后，冻结合法原决策全集及 unknown；不依据错例、chosen y 或目标标签挑子集。

在随后另行准入的最小测量中，同一已许可来源／原 R 只需要每个固定 neighborhood 的一次原序列 teacher-forcing，加完整来源基准（可复用同模型／精度／原轴已资格的缓存）。一份输出同时产生 uniform mixture、fixed-prior predictive、Bayesian prefix predictive 与过去 prefix 的 argmax reranker；不再各跑一个模型，不在观测后选赢家。若来源有 K 个专家，完整物理预算是 K 个额外序列前向，非一个；成本需保存 Σ_j(P_j+T−1) 输入位置，KV 重复的 R 历史及 loader 峰值另计。

预先竞争预测：

- **关系身份假说：** 原 prefix 已闭合 owner/action 后，source licence 区分的相容事实应在值出现前获得较高 posterior；将完整两事实的 owner/action 身份对调而保留 value multiset，应按许可方向交换该偏好。来源事实整段保义重排只允许位置控制；不可认为任意 shuffle 保义。
- **词频／回声假说：** 只复制一个同义事实 witness、或换成同主题但 owner 不同的邻域，就能造成与主体交换相当的 posterior 转移；或 posterior 要等看到原 y 后才偏向可解释 y 的专家。这些结果否定所需的新判别，而非支持“模型在回看”。
- **正常切换控制：** 原 prefix 从第一事实明确转向第二事实时，应允许相容专家变化。若 switch 大小同时给正确切换强警报，不能把 switch 单独当 risk，也不能靠重切 R／调转移温度救。

必要门须在新值前冻结“相容事实集合”以及 identity/synonym/重复/重排的预期，且事实合法算术不能被当作错误。当前没有这个自动 source 许可器；手工有限许可只支持必要能力研究，不能宣称自然 coverage 或新 detector。强直接文本基线使用完全相同许可集合与成本账本；在显式 owner/value toy 上若规则已完美，内部概率仅能提供能力资格，不产生检测收益结论。

**决策：** generic latent-fact mixture／posterior-switch detector **NO-GO**；新实验仍只是上述 source-identity-before-y 的必要判别待审。若连 source-only 完整邻域、同义重复类别与 prechoice owner 资格都不能冻结，就停止新增模型调用，而不是改名运行 P021／H005。若必要能力通过，下一门才是原 R 上的 coverage、定位、成本与同信息强文本核验增量；proper-mixture 恒等式本身不满足任何一项。
