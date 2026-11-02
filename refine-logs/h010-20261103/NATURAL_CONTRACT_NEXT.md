# 自然重建前端：可识别性、原问题来源与下一项最小桥接

2026-11-03。状态：研究合同与主文献审查，不是已实现/已运行的自然检测器。应用本地 research-pipeline 的证据分层与独立反驳原则、experiment-plan 的 claim→evidence→run-order、最多两个切入口和明确失败解释；按本次限定只写此文档，不启动完整 pipeline、模型、QA、下载或状态变更。本轮只读 H010 工程摘要、既有问题合同审计/格式与公开主文献；没有打开 evaluation_only、benchmark gold、原未编辑回答或本地预测数值来选择模板。

**结论：H008 通过了一个必要但远非充分的来源 QA 能力门。自然迁移的缺口是“问题究竟把哪个关系参数当作可变、其余锚点是否可靠”，以及来源候选是否完整。不能把名字遮蔽、逆向 QA、两指派归一化本身当作新方法。建议下一步仅做下面固定的 H011 已知槽位桥接；136 个自然 NER occurrence 暂不进入效果结论。**

## 1. 反例成立的对象，以及不能推出的结论

考虑

\[
S=\{\operatorname{wins}(X,A),\operatorname{wins}(Y,B)\},\qquad
R=\operatorname{wins}(X,B).
\]

有两种被来源直接支持的修复：把人 X 改为 Y，得到 wins(Y,B)；或者把事件 B 改为 A，得到 wins(X,A)。若潜在世界 W_person 的原意是前者、只损坏 person，W_event 的原意是后者、只损坏 event，它们可给检测器完全相同的 S、R。没有原问题、编辑来源或规定哪个角色不可改变的锚，任何仅依赖 S、R 的函数都不能保证在两世界中同时定位唯一的“被改坏参数”。这是**编辑来源/意图的最坏情形非识别性**；不是所有自然文本的错误位置都无法预测。

需要反驳一个容易被省略的前提：若 S 只是开放世界中的两个正事实，S 不蕴含 wins(X,B)，却也未必蕴含其否定。要称其“矛盾”，还需 sole winner、事件唯一性、穷尽性或闭世界等额外公理。没有这些公理时，安全表述是来源未支持。即便加入严格双射使 R 确实错误，以上两种编辑来源仍然不可区分。

可识别的对象仍然可以是整个关系不被支持、一个关联参数集合有问题，或所有最小支持修复的集合。输出 person 与 event 的联合疑点、或整个关系风险，并不违反反例。真正不能保证的是从这对输入恢复唯一的因果编辑标签。原始生成问题若明确问“谁赢了 B”，可把 B 规定为题意锚；若问“X 赢了哪个事件”，可把 X 规定为锚。宽泛问题、错误问题或多答案问题则未必消除歧义；原问题也不揭示生成器内部因果。

这个论证**不推出全局 AUROC=0.5、不推出任何方法不能达到 0.8，也不要求放弃 token 评价**。在一个额外假设“两个编辑世界等概率且标注恰好相反”的局部二选一任务中，确有相应的不可兼得界；但真实语料还含可识别样本、非对称编辑分布、上下文和稳定标注规则。更直接地说，若只在占全部正负比较对比例 α 的一部分上表现如随机、其余比较对全对，一个示意性 AUROC 就是 1−α/2；它并不是恒等于 0.5 的全局上界。该式只是反驳非法推广，不是在估计本数据 α。

目标必须先声明是来源蕴含、最小修复、编辑出处还是数据集规定的 span。RAGTruth 是人工 word/span 标注语料；FRED 的 span 来自受控标记编辑，它们的定位对象不能自动视为同一潜变量。标注策略即使不能反映唯一生成原因，也可以定义一个有效、可预测的 benchmark 目标。[RAGTruth 原论文](https://aclanthology.org/2024.acl-long.585/)、[FRED 原论文](https://arxiv.org/html/2507.20930v2#S2)。

## 2. R→Q 遮蔽能防什么，不能证明什么

对于上述反例，遮住 person 得到“谁赢了 B”，S 回答 Y，与 R 的 X 不同；遮住 event 得到“X 赢了哪个事件”，S 回答 A，与 R 的 B 不同。两条 mismatch 都可正确提示关系不兼容，却不能据此声称找到了真正被编辑的那个 token。每个问题都以另一项可能错误的 R 值为条件；目标值不在问题里，不代表其余条件已是真实锚。

H008 不需要解决这个问题：它先从已知严格来源语法确定事件角色，用 S 决定 Q，再评分完整人名+EOS。程序还已经理解本 toy 的关系绑定。自然 R→Q 必须另证目标角色保留、断言范围、指代、问题唯一性和来源可答性；NER offset 准确不是这些性质的证明。

H010 已有前端记录 18 答、36 字段，source 719 与 R 136 个已输出实体 occurrence；未知语义可答率/支持率没有被评估。spaCy NER/POS/dependency 管线依赖外部 OntoNotes 等监督，不能称作完全无监督训练的前端。它没有使用本轮目标标签；必须同时保留这一区别及类型/别名错误的可能性。[本地 FEASIBILITY](FEASIBILITY.md)、[官方英文模型说明](https://spacy.io/models/en)。

| 合同问题 | 必须保留的边界 |
|---|---|
| 原值擦除 | R 中目标完整字面值的所有 occurrence 都擦除，包含 NER 漏标的字面重复；原值不能经生成问题、隐藏前缀、同义括注、上下文或请求说明重新拼回。S 原文包含候选是证据，不是 R 值泄漏。 |
| 同类型值擦除 | 避免另一 PERSON/数量直接给出待填值，但多变量同时擦除会毁掉唯一性。占位符须保留 occurrence/角色区别；不能把“更干净”自动等同“更可答”。 |
| 别名与类型 | PERSON/ORG、GPE/LOC、CARDINAL/DATE 混标、简称、词形、重叠和跨句指代可能留下信息或拆开同一实体。字面没有残留只证明字面门，不能证明语义无泄漏。未冻结的别名/类型合并规则不得按案例效果增补。 |
| 来源候选域 | 候选仅由 S 独立产生；R 原值、编辑前答案、标签或模型风险不得用来保留/删去/优先候选。表面同类型实体不等于完整答案域；无法证明完整时保留 domain-incomplete/unknown，而非把域外正确答案当错误。 |
| 合法算术 | S 的数值表面池不能覆盖 opening=−34.8、closing=46.0 所蕴含的净变化 80.8。闭包必须按已锚定运算、单位、时间方向、尺度和操作数出处定义；“可算出一个数”不等于该谓词问的就是它。不得从 R 的答案或 gold 反向挑运算。 |
| 缺失与非断言 | 来源缺证与冲突分开；未提及事实与回答遗漏分开；问题、请求、意见、条件句、引述和否定作用域另保留。不能让 source-recall 的缺漏自动变成 response-hallucination。 |
| 未知 | query-ambiguous、nonassertive/unresolved、alias-unresolved、candidate-incomplete、budget-failure 分别记录。数值 unknown-risk=1 若作为保守工程回退，仍不是“已预测矛盾”，不删除困难原答/原 token。 |

自然任务也不能因为恰有两个名字与两个事件就套双射。H009 的显式 source-is-admitted-bijection 前提是额外语义合同；有限候选归一化既不补全候选，也不把能量和变成独立事实似然。

## 3. 主文献核查：哪些思想已经存在

本节只引用作者论文/官方发布，未使用二手评测或它们的 benchmark 结果选择本地组件。下列重叠足以排除“逆向 QA/NER 遮蔽/细粒度核验是新思想”的说法；它不构成穷尽的新颖性调查。

| 主来源 | 已有机制与本轮边界 |
|---|---|
| [QAGS，ACL 2020](https://aclanthology.org/2020.acl-main.450/) §3–4 | 从生成文本产生问题，在生成文本和来源上回答并比较；还用回答一致性过滤问题。R→Q→source QA 与问题有效性门已有直接先例。它不保证被定位的词就是唯一原编辑位置。 |
| [FEQA，ACL 2020](https://aclanthology.org/2020.acl-main.454/) §3/Fig.2 | 明确遮蔽 summary 的实体/名词短语，生成对应问题，再与来源抽取答案比较。本文所谓原值擦除不能仅凭“使用 mask”宣称创新。论文中的 summary-derived answer 不等于 benchmark 真值。 |
| [Q²，EMNLP 2021](https://aclanthology.org/2021.emnlp-main.619/) §2 | 用 spaCy 实体/名词短语、QG、source QA 和答案 NLI；验证问题在 R 中可恢复，并处理个人/意见内容及无有效问题情况。NER+QA 和别名/非断言问题均有先例，外部监督与过滤覆盖也须披露。 |
| [QuestEval，EMNLP 2021](https://aclanthology.org/2021.emnlp-main.529/) §4 | 区分从生成文本提问的 precision 与从来源提问的 recall，兼顾答案/可答性；还有学习的显著性权重。不能把来源事实未在 R 出现直接当幻觉，也不能把 reference-free 误写成无外部监督。 |
| [QAFactEval，NAACL 2022](https://aclanthology.org/2022.naacl-main.187/) §3–5 | 分析答案选择、问题生成、回答、答案匹配及可答性；指出 QG/answerability 至关重要。其经验证集组件比较和 learned 合并不自动符合本轮禁止目标标签选择的合同，不能照搬为本轮“无监督最优权重”。 |
| [RefChecker / Knowledge-Centric Hallucination Detection，EMNLP 2024](https://aclanthology.org/2024.emnlp-main.395/) §3/Limitations | 从 R 提取 claim-triplets，按参考判别并聚合，保留 entailment/neutral/contradiction。作者也明确三元组对时间、复杂推理、表格等覆盖有限；细粒度关系提取与核验不是本轮新原理。 |
| [Kuhn，1955](https://onlinelibrary.wiley.com/doi/10.1002/nav.3800020109) | assignment 是经典优化问题；2×2 只有两种可行指派，枚举其和分数/MAP 不是新算法。行列加性项对两指派加相同常数的消去是代数性质，不是自然姓名偏好必然可加的证据。 |

更窄的“所有目标/同类型值擦除并给出可识别性与候选完整性失败清单”可以研究其作用条件；目前只能称合同/诊断设计，尚未证实其不同于已有方法的实质贡献，更不能把 H008 toy 100% 或 H009 score-nuisance 不变性包装成自然机制。

## 4. FRED 原问题：已知缺失不等于所有合法恢复入口均被排除

复用既有封存审计，不重跑：outputs/research-goal-20261102/dataset_contract_audit.json，SHA256 为 17c4cc1d828980c309f793c87d93f9e62765b5d9f8684b411a25f55615f590e1；其 audit seal 匹配。tasks.FRED_FinQA.question_parse 和 tasks.FRED_TATQA.question_parse 的 parsed_responses/distinct_questions 都为 0。原组分别 162 答/142 个 reference source 与 130 答/110 source；source_id 是 exact reference text 的 SHA256，不是原 QA id。format-check 的 SHA 为 b687be2452301bf51c4fd42450ef0ab449a948bc65268d610d2dc6b89f5033aa，其 seal 已核对；本轮只读这些既有结论。

必须更正恢复路线的起点：FRED 官方论文 §2 说 FinQA/TAT-QA 来自 **RagBench**。[官方论文 §2](https://arxiv.org/html/2507.20930v2#S2)。作者历史 revision afdab2c54075ade8c890331da099cc9b25d1f6a5 的 [convert_format.py](https://github.com/pegasi-ai/fine-grained-editting/blob/afdab2c54075ade8c890331da099cc9b25d1f6a5/data_preparation/convert_format.py) 使用 documents[0] 为 evidence，最终只导出 completion/prompt；脚本 SHA256 99dd66cb4e52a3a07319d78393ecd644958c0f9bdeed2f78c56a06f78c51a6e2。两份 test CSV 的 header 都是空 index、completion、prompt，没有 question/native id。不能把无名行号当作 RagBench 或原 FinQA/TAT-QA ID。

[RagBench 官方固定 schema](https://huggingface.co/datasets/galileo-ai/ragbench/blob/97808f3e5fd16ede40bbff6c2949af8139b2eb7b/README.md) 的 finqa/tatqa 有 id、question、documents，数据许可 CC-BY-4.0。[FinQA 官方 schema](https://github.com/czyssrs/FinQA#dataset) 有 entry.id 与 qa.question；[TAT-QA 官方 schema](https://nextplusplus.github.io/TAT-QA/) 有 table.uid、questions[].uid/question，且一个 context 可有多个问题；[TAT-QA 数据许可](https://github.com/NExTplusplus/TAT-QA#license) 为 CC-BY-4.0，不能与代码 MIT 混为一谈。但这些上游 ID 没有在当前 FRED 最终 CSV 中保留，原生库中“存在 question”不等于已经完成合法精确回连。

**尚未执行、也尚未被排除的合法入口**：另行冻结只读取 RagBench 的 id/question/documents 三列，按官方 converter 的首 document 及已冻结本地 reference 提取规则，做 source-only 全基数 exact join。每个 FRED source 保留所有匹配 upstream row id 与所有不同 question 字节值；只有不同 question 唯一时才可标记 Q-recovered。无匹配或多不同 Q 均 unknown，不猜测；不使用被编辑/编辑前回答、completion、标签、正确答案、程序答案或风险分数消歧。不做模糊文本阈值搜索来追求匹配率。

本次只核 schema/header/历史代码和已有 audit，没有运行新 join，没有下载上游大 payload，也没有读取其 answer/response/评价字段值。即使未来问题可恢复，也不意味着恢复了原完整生成 prompt、全部检索 documents、文档顺序或消息角色；只取首 doc 的既有 FRED 观测仍是不同上下文合同。故当前结论应是“直接 ID 恢复不可用；RagBench 首 doc 唯一问题回连待审”，而不是“原问题永远不可恢复”。不可把原未编辑 response 当输入补 Q。

## 5. 最多两个切入口

**C1（当前主入口）：回答值盲化的角色问题能否跨接口迁移，并保留可识别性失败。** 反主张是“来源角色已被 toy parser 解出，R 遮蔽后题面仍不可靠，名字或错误的其余条件决定了结果”。最低证据是固定、完整、值不泄漏的角色查询在必要桥接上成立，且能拒绝不唯一锚；不是仅显示一组 QA 正确。H008 只支持此前半项必要能力，不支持自然前端已合格。

**C2（后续条件入口，不与 C1 同时扩张）：问题/证据来源锚与答案域完整性。** 对可合法恢复的原 Q 或可验证角色锚，明确单位/时间/运算后再考虑来源候选与有限算术闭包；无锚/别名/缺候选明确 unknown。反主张是“表面不在 S 的合法推导被当作幻觉”。最低证据包括原问题的无泄漏 provenance、合法推导保留与显式域外失败；不把 NER 匹配或任意表达式枚举作为支持真值。本轮不实现此入口，也不将它拼成多权重混合。

## 6. 唯一下一项最小可证伪实验：H011 已知 person-slot 的 R-masked-Q 桥

这是 root 最新提出的桥接，不是已经完成的结果，也不是自然 136-span 评分。H011 保留 H007 原 48 输入、严格已知的两个 PERSON 原槽位、两原事件和全部 0/4/12 后缀，隔离 NER 漏检/类型/别名因素。只检验 H008 的 S-defined question 能力能否迁移到由 R 擦除两人名得到的题面。

**可识别性准入**：来源已被独立固定语法验证为两个不同事件/两个人的一对一绑定；问题预先规定“只问 person，两个 event 条件不可修改”。对一个 target 槽，在完整 R 中擦除其人名的全部字面 occurrence 为 [TARGET]，把另一人名的全部 occurrence 擦为 [OTHER]；保留原事件条件、标点/空白和后缀。不允许把被擦掉的原名字经 target description、其他消息或 Q 再塞回来。原值仅保留在模型不可见的原槽对齐/评价映射；候选清单仅来自 S 的两个人名，规范顺序在读 R 值前固定。

固定唯一指令，不进行替换/搜索：

    Using only source, give full person name for [TARGET] in masked statement; [OTHER] is a different source person.

其中 “[OTHER] is a different source person” 是**显式 toy 双射知识**。它不是 R 原值泄漏，但确实缩小了合法解域；不能把这条额外条件默认为自然世界真理。该桥选择 person 作为待填参数、event 作为固定锚，因而不与第 1 节的无锚编辑来源反例矛盾。它得到的是给定事件条件的 person 一致性，不是从一般 S、R 唯一找回原编辑 token。

两种 R direction 的全人名擦除后，应在同 S/负担/target-slot 下得到**字节完全相同**的模型题面。每原答保留两个目标与两候选，所以是 48×2×2=192 个逻辑 completion 实例；经精确复用后为 8 sources×3 burdens×2 target roles=48 个唯一问题，至多 **96 次**完整人名+EOS 前向。差异原答和重复实例仍逐一保留，不按唯一题面丢掉原记录。不把 protocol/version hash 变化当成重跑相同数值请求的理由；冻结时核查消息、候选、完整 token framing、模型/数值合同及现有缓存，能够严格复用的请求必须复用。

模型、runtime、量化、注意力、完整姓名+固定 EOS、每 token 全词表 float64 normalizer 和 joint sum 沿用已合格 H008 的数值合同；无采样、无新长度/温度/任务权重。候选原名作为 teacher-forced 被评分文本是观测本身；它不在问题前缀中以“这是 R 原答案”的身份被告知。先完成全部 192→96 映射与所有 metadata 的无模型 preflight，冻结源/模型/协议/问题清单后才允许调用。任何不坍缩、名字残留、事件/后缀改变、缺候选或边界不稳应工程失败且 0 个实际 forward；不可运行后修题再混并。

| 固定项 | 合同 |
|---|---|
| 主问题 | 同一个已知事件角色经完整 R 遮蔽+后缀承载后，是否仍能来源重建？ |
| 唯一必要对照 | 复用 H008 已封存的 S-defined 16 QA/32 completion，全结果保留，不新跑对照。H007 仅作不同接口的既有背景，不称同预算。 |
| 主读出 | 全部 48 唯一问题按 0/4/12 分层的 greedy 正确、wrong、精确 tie、unknown；各层全 8 sourceflip 与全 8 eventflip。完整 joint LL、pool mass、长度/成本逐一保留。 |
| 次级读出 | 全部原 48 回答/96 槽的固定风险与 AUROC/AP，仅在预测与 CPU evaluator 冻结后使用原 construction 参考。不是自然 token 实效，也不是编辑来源正确率。 |
| 证伪与停走 | 0 负担若不能恢复 H008 已合格的 16/16，说明 S-defined→R-cloze 迁移失败，不能说基础模型不懂来源；不换 prompt 救分。若 0 负担通过而 4/12 下降，支持此 QA 接口也受后缀影响，仍不能唯一归因为某个内部机制。若全通过，只进入自然锚/候选/别名合同审核，不晋升为自然检测器。 |
| 预算与冻结 | ≤96 个新增 completion forward、0 新模型/大下载；真实时长待执行记录，不能把 H008 4.13 秒当本轮承诺。全部逻辑实例保留，无 prompt sweep、无参数选择、无改标签。 |

这项新信息区别于 H007 的 A/B/C verdict 和 P021 的原回答 continuation likelihood；它也不是重述 H008 的 source-defined Q，而是将同一必要能力放进 R-derived masked cloze 与受控后缀的新条件。它仍是已知 toy 角色的桥，不是自动自然 R→Q 提案的验证。若文档/实现把 person-slot 合同、不同人约束、完整原事件或 source-only 候选任一项悄悄移除，96 请求的可识别性与计数结论都必须重新审查，不能沿用本准入。

自然 18 答的 136 个 NER 提案随后至少要保留：目标/同类型字面擦除图、其余可能错误的条件、断言类型、唯一/多锚/无锚、候选来源/完整性与 unknown 原因。未完成这些检查前，不把 occurrence 数量写作语义覆盖；也不使用旧自然分数或标签来挑出“适合问”的 span。

## 本次交付边界

无核心/状态文件修改，无模型/QA/GPU 调用，无上游大下载，无新数据 join，无 evaluation_only 或预测数值选择。主文献和 schema 阅读截止本地 2026-11-03。FRED 子审与非识别性论证均有独立只读反驳；H011 的条件性准入也经独立检查。所有推荐是前瞻合同，H008/H009 的已封存结果与既有限制不被覆盖或重新命名。
