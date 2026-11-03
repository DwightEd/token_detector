# H017 关系错绑：面向当前实验的文献补充审查

调研日期：2026-11-03。范围：公开论文及作者资源、当前 H017 计划、指定历史实验和实现合同。本次未读取目标标注、内部数组数值或运行模型；文献结论不构成本项目的检测结果。本文件补充既有 [绑定综述](20261103_entity_binding_review.md)，不覆盖旧证据。

**当前最有依据的切入口是检索地址、事实所属主体/条件、值传输与最终读出之间的分离。** 不能把“来源注意力强”“隐状态相似”“读出稳定”直接变成关系正确性，也不能把注意力特征本身称为新贡献。可检验的新信息应是：固定值和位置时，内部计算是否随来源的角色归属而变；这种变化能否区别位置、字面重复、回答回声和合法多事实计算，并改善原回答检测或定位。

## 一、原始研究支持什么

| 原始来源 | 经核实的结果及对 H017 的约束 |
|---|---|
| [Feng & Steinhardt：How do Language Models Bind Entities in Context?，ICLR 2024](https://arxiv.org/html/2310.17191v2) | 用激活交换、位置控制和向量干预支持 binding ID 的可分解表示；也报告不使用该机制的任务。依据是特定表示和干预，不是完整原始隐状态的 cosine。实体后的接续 token 也可能携带绑定。H017 应预定 occurrence/接收位置，不能只看名字自身或把所有自然关系假设成一一对应。 |
| [Oh & Demberg：A retrieval conditioned rebinding circuit…，EMNLP 2026](https://aclanthology.org/2026.emnlp-main.1238/)；[方法全文 §4–6](https://arxiv.org/html/2606.08644v2) | 研究动态换位后的实体追踪：存在检索时重绑定的紧凑注意力路径；Gemma 的 Q/K 匹配与 Llama 更偏 K 的实现不同。用地址变化与新内容变化的不同输出目标区分 routing/content，再干预 Q、K、Q+K。**更新旧综述的发表状态：已核实 2026 年 10 月 EMNLP 正式记录。** 它未证明当前 Llama3.1、block15、自然回答错例存在相同电路。 |
| [Chuang 等：Lookback Lens，2024](https://arxiv.org/html/2407.07071v1)；[作者代码](https://github.com/voidism/Lookback-Lens) | 每个 head 的 context/生成历史注意力比例进入有监督线性分类器；不是无监督正确性定理。原研究含跨任务/模型迁移。它已经研究“看来源还是看自身生成”，故 H017 不能只换一个来源总量或比例。若作为冻结外部基线，须有匹配观测的固定版本，不在目标标签上重训，并披露训练数据暴露。 |
| [Matys 等：AggTruth，ICCS 2025](https://www.iccs-meeting.org/archive/iccs2025/papers/159110220.pdf)；[方法全文 §3](https://arxiv.org/html/2506.18628v1) | 来源注意力的总量、head 间 cosine、entropy、JS 分布特征及有监督 head 选择已有先例。§3.3 特意区分预测当前 token 的行和读入该 token 后的行：后者能描述已生成 token，前者不受该 token 的实际抽样结果影响。H017 必须分别定义检测信息合同与生成因果合同；不能把 post-token 信号当成事先已有的机制。 |
| [Ogasa & Arase：Hallucinated Span Detection with Multi-View Attention Features，*SEM 2025](https://aclanthology.org/2025.starsem-1.31/)；[方法全文](https://arxiv.org/html/2504.04335v1) | 平均 incoming attention、incoming/outgoing entropy 加 Transformer/CRF 做有监督 span 检测；attention observer 可以与原生成器不同。其 norm-adjusted attention 也考虑 V/O 变换。故“额外 observer 内部特征”和“局部集中度/输出历史特征”均已有近邻，冻结 LLM 不等于整个检测器无监督。 |
| [Sun 等：ReDeEP，ICLR 2025，作者页面](https://jeryi-sun.github.io/ReDeEP_Page/)；[论文方法 §3–4](https://arxiv.org/html/2410.11414v1) | 分离外部上下文利用与 FFN 参数知识：attended 来源状态相似度及 logit-lens/FFN 分布变化，并做路径贡献调节。因此“来源与内部知识竞争”不是 H017 的新发现；单层 QKV 不足以原样实现其完整多层读出。H017 应针对来源内部的关系归属，而非把所有错误归为未使用来源。 |
| [Yamada & Arase：Hallucination Span Detection with Input-Side Evidence Alignment，2026-08 预印本](https://arxiv.org/html/2608.15804v1) | 用 RAGTruth 正误标签训练 masked-span 来源检索及置信度，随后用开发标签调阈值；不是当前允许的无监督训练。§6.2 明确发现 conflict 难例：相矛盾片段也高度相关，因此相似度高仍会误判。**“能找到证据片段”和“该片段支持回答关系”是不同问题。** 此文仅作为近邻和失败机制证据，不能搬其目标标签训练协议。 |
| [Hu 等：RefChecker，EMNLP 2024](https://aclanthology.org/2024.emnlp-main.395/)；[Tang 等：MiniCheck，EMNLP 2024](https://aclanthology.org/2024.emnlp-main.499/) | RefChecker 已有 claim-triplet 抽取再核验；MiniCheck 用外部合成训练构建较小的 grounding verifier。它们是直接文本核验的必要近邻。三元组拆分和核验不构成独立新意，文本基线也不能只选一个受限 NLI pilot 后宣称白盒更强。其论文成本/精度不能直接移植成当前数据和设备上的结果。 |

上述工作共同支持“检索了什么、怎样传值、最终说了什么分别检验”；没有论文保证 source mass 越大或 owner-profile 越稳定就越正确。归属和条件必须包括谓词、时间/范围、否定与原始 occurrence，数值词袋相同不许可语义交换。

## 二、为什么粗粒度观测会漏掉这类错误

以下是可检验的推断，并非已测的 R4/V3 不变性。来源为“甲100、乙200”，回答固定为“甲200”：整体 source mass 可能很高，且“200”的定位非常集中，因为错误可能发生在地址/所属条件，而非值是否出现在来源。对来源 token 分布求和会丢失地址，熵会丢失身份；相似度可能保留“收入”话题而丢失“哪个公司/哪一年”。因此这些标量对某些角色交换可能不可识别，须实际测量其响应，不能预先断言所有基线均失败。

注意力本身还只指定取值权重。局部 head 输出为

\[
o_{t,h}=\sum_{s\le t}a_{t,h,s}v_{s,h},\qquad
a_{t,h,s}\propto\exp\!\left(\widetilde q_{t,h}^{\mathsf T}\widetilde k_{s,h}/\sqrt{d_h}\right).
\]

高权重不决定 V 中携带什么，也不决定 O 投影、残差、后续 MLP/层及 unembedding 怎样使用它。错误回答在读入后还能形成一致的历史，故稳定性可能是回声。当前未保存 O 权重或完整后续网络，局部 AV 变化不能冒充最终输出因果。

## 三、三个有区分力的预测

这些是补采/分析前应冻结的竞争预测；不是新的检测分数。自然标签最后评价，不按它们挑 head、层或效应阈值。

1. **归属敏感与位置/字面敏感可以分离。** 在独立、语义许可的控制里保持数值 occurrence、位置和回答前缀，真实交换主体或条件归属；另做完整事实的保义重排、仅字面变化、重复合法值。若观测反映归属，它应跟随事实身份，并在保义重排映射回原 occurrence 后保持；若只跟随 token 位置/数值字符串，则拒绝归属解释。纯 CPU 重标同一数组不能通过此门。先声明效应/等价界，保留全部32物理 head；来源质量或 offset 不够时 unknown。

2. **K 地址、Q 请求和 V 内容具有不同干预预测。** 用固定值/位置的角色交叉作为 donor，分别替换接收端 Q、来源相关 K、对应 V，并做一个预定交互。地址机制预期：在原 V 保留时，Q/K 改变能重定向 attention，最终 logits 朝原输入中被新地址取到的值变化；内容机制预期：V 改变影响取出的内容，而不必产生相同的地址变化。K 主导是 Llama 文献给出的优先竞争假设，Q+K 抵消并非 Llama 必过门。须同时有 no-op、无关事实、匹配范数随机方向、正确案例损伤和保义重排；全向量替换还可能带入其他上下文，不能自称纯 binding-ID 操作。只有同一冻结 observer 的完整续算才检验最终输出，cache-only 重放仅是局部计算。

3. **早期条件失败、检索切换、读出失败和回声不能压成同一个“漂移越大越错”。** 前缀条件已可得时，比较条件接收行、prevalue 行与 postvalue 行。条件阶段偏甲、prevalue 偏乙支持检索切换候选；两处都偏乙支持条件读取/表示失败候选；两处都偏甲却输出200支持值/后续读出失败或当前层观测不足；仅 postvalue 稳定支持回声候选。合法跨事实计算可有大漂移，错误也可始终稳定。若主体/条件尚未出现且原问题也未提供，不能偷看后文解释原生成，预选择分析为 unknown。当前只有一层，这些首先是关联分类，需预测2的干预区分。

这些实验操作原回答前缀的内部处理和原 token/candidate logits，不另建独立角色 QA 成绩榜。当前 cache 缺来源读取阶段的 Q/hidden，所以不能据此声称已经找到“读完来源后整理关系、再回看核验”的节点。

## 四、实际缓存与基线的接入边界

- 当前 QA900 有 block15 的 pre-RoPE Q/K、完整输入 K/V、回答端 T+1 Q/nodes；必须先通过真实 RoPE/GQA/causal/world-mask 重放资格。阻断世界实际影响执行的0–15层；它不是某一层角色路径干预。early-stop 没有最终 logp，不能把 logit lens 或缺失字段补称高生成概率。
- 原回答来自多个生成器；Llama observer 对已有回答 teacher-force。当前机制结论最多先指向 observer 怎样处理这个回答，不是 GPT/Mistral/Llama2 原采样原因。增加 observer 的时间/显存/缓存 I/O 必须计入端到端费用。
- [当前主线](../../myproject/shared.py)、[Detector](../../myproject/detector.py) 继续给出原 R4/V3 比较，保持原轴。[attention_profiles](../../myproject/attention.py) 是 post-token 且跨层/parity 聚合，不能作为 block15 的32头 prechoice 证据。固定总 source/value mass、集中度作为单量对照，不按目标表现加权。
- [现有 NLI](../../experiments/nli_qualification.py) 固定 DeBERTa revision、外部监督、512 pair 窗口及 `1-max entailment`，但其 runner 强制34例/104 units，省略原问题/回答上下文。可复用经过审查的 backend/身份/seal，**不能直接调用旧 pilot 充作 QA900 的全量强基线**。扩大人口或上下文需新合同，不能悄改旧已封存实验。
- 若进入检测比较，先独立固定一个可获得的较强 grounding verifier（例如 MiniCheck 发布版本），不按目标标签筛选。内部与文本方法共享事实候选/可用上下文，并另报各自前端端到端覆盖/成本；保留简单结构规则对照。RefChecker 式抽取本身有误差和调用成本。公开权重若接受过目标数据训练，不得称目标无监督，应另列外部监督/污染边界。
- 主任务输出仍是原回答的风险和原 token/span 轴。错误边“甲—200”可通过改主体或改值修复，不能仅凭概念例子指定唯一坏 token；边→跨度投影、unknown政策和全部分数须先冻结，随后才评 AUROC/AP、定位、固定预算 FN/FP 和全分母覆盖。

## 五、与已关闭实验的实质区别和停止条件

| 已关闭路线 | 本轮只有增加什么才算不同问题 |
|---|---|
| P002 原始锚点残差/cosine | 预定 role-sensitive 物理 Q/K 路径、语义许可的真实角色交叉与保义对照；再换距离/正负号不算。 |
| P003 删除依赖指纹 | 地址与内容分离的干预预测及完整输出终点；删除影响仍不能证明绑定正确。 |
| P005 post-token 方向/whitening | 明确 prechoice 与 post-token，事实所属主体/条件而非 source-induced 全局方向；不能重报历史选例。 |
| P021 数字行列 null/source worlds | 独立许可的角色交叉、位置/值控制、Q/K/V 路径；保值词袋和普通 shuffle 本身不建立语义有效性。 |
| P026 repair/reset | 当前原前缀路径检验；不再用单位 reset、年替换及未来概率重复测试。 |

若仪器未合格，停止解释机制；若来源事实无法许可，保留原人口和 unknown，停止该域语义评分；若 role-sensitive 量只反映顺序或字面，拒绝该读出；若内部机制可解释却不改善同预算文本核验的检测、定位或成本，就保留机制负/局部结果，不发布为更好的检测方法。目标仍是六任务 AUROC≥0.8，文献与本次文档没有完成该目标，也没有建立新颖性或顶会水平。

本次科学审查依据：[当前计划](../../refine-logs/EXPERIMENT_PLAN.md)、[H017准入](../../research-state/relation_misbinding_admission.json)、[历史去重索引](../../research-state/prior_experiments.json)。M0 的工程范围已独立审阅：只许可原始结构路径/标量 occurrence，prose 关系保持 unknown，测试须覆盖 Unicode 字节/字符 offset、重复值、重复键拒绝与跨边界 token；这不等于自然语义前端或检测器已准入。
