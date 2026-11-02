# H002：先检验接口与可观测性，再讨论检测方法

状态：根代理已选择 B/H002 进行 **qualification**。这还不是新主线、新贡献或检测结果。A 的条件执行器属于已有能力与先验控制，不再把人工已知错例的重演称为新方法 pilot。当前只实现纯输入／输出合同和合成回归测试；没有新模型调用，没有新金标读取，没有阈值、权重或 token 分数。

## 真实的不确定性边界

旧方案让看过待核值的提议器生成问题，即使字符串检查显示问题没有包含该值，也不能证明问题没有携带答案信息。当前合同不接受模型生成的问题。

提议器只接收**完整、未标亮的原回答**，自动给出全部事实 claim、typed slot 及原字符文本；它看不到来源、金标或人工挑选的位置。唯一的 exact claim text 在完整回答内确定性对齐，slot text 在该 claim 内唯一时对齐。重复文本没有有效显式位置就记 invalid；显式位置错误不回退猜测。默认不要求小模型逐字符计数。提议截断、坏 JSON、空 slot 和未覆盖内容都属于覆盖／接口限制，不能解释成语义假设被证伪。

给定提议的 claim 与 slot，将所选值、其他字面重复与声明别名确定性替换成固定标记。三个模式共享同一自动提议：

| 模式 | 实际观察输入 | 作用 |
|---|---|---|
| direct | 来源 + 完整 claim + role + 自动 focus span | 强直接核验对照，主动看到声明值 |
| blind | 来源 + masked claim + role | 不传原完整回答、值、别名、offset 或生成的问题 |
| leaky | 与 blind 完全相同，再显式加入 claimed value | 暴露声明值的对照 |

设来源为 \(S\)，掩码后的 claim 为 \(C\)，角色为 \(R\)，序列化合同为确定性函数 \(B(S,C,R)\)。只替换 slot 值且保持 \(C,R\) 完全一致时，两次 blind 请求的字节相同。合成回归测试验证的正是这个**条件输入隔离**：例如保障期 annual／monthly 改变时，开票 annually 保持原样，blind 输入一致，而 direct 与 leaky 改变。

这个结论不保证上游 \(F(Y)\) 提议的 frame／slot 与值独立，也不排除未声明别名、语义暗示或提议选择偏差。字面检查采用 casefold、空白规范化与词边界，只是有限的查询侧检查。来源本来可能包含该值，**不得为了“盲”而删改或屏蔽来源**。

回答必须是 answerable／unknown，引用需字面唯一，或显式位置严格匹配。引用成员关系不等于蕴含、来源可靠性或模型判断正确。不同实体字符串也不自动判矛盾；别名、缺失比较前提、角色歧义和数值执行需求应保留 unknown。这里没有算术执行器，不把正确数字当成正确账户绑定的证据。

## Q0 是先决资格检验

原始主评估 answerable891 条中，567 条 Llama2 回答全部带前导空白，324 条 Mistral 回答全部不带。该混杂来自 `dataset_contract_audit.json` 的 `cross_tabs.Hallu_primary_generator_leading_whitespace`，不能外推到全部1782条，更不能由此断言空白导致误报。

纯工具对每条完整回答始终保留五个 arms：原文、增加一个 ASCII 前导空格、仅在已有 ASCII 前导空格时删除一个、在合适内部空格处增加一个、仅有可冗余内部空格时删除一个。不可执行的控制仍保留 identity arm、空删除列表与明确 reason，不筛掉样本。Unicode 空白、换行和 tab 不被整体 lstrip。每个变体都有原字符映射，映射入口重新检查范围、顺序、重复、文本和实际删除位置。

这些工具只证明字符顺序与合同验证，不证明 LM／接口因果效应。`token_count_matched=None`、`unit_format_matched=None`、`semantic_equivalence_verified=False` 被显式保留。代码、表格或引号中的空白可能带有意义；实际 Q0 还需固定 tokenizer／后端，报告分词、unit 和来源 gap 的变化，并用可行的内部空白控制区分格式与语义因素。不得删除原首几个 token，不改变原总体，不将未知标签变成负例。

## 固定错例应检验什么

本表是预期和反证条件，不是模型已给出的答案。方法输入仍是原始完整来源与原回答，不能提供本表人工选出的错误值或金标位置。

| 已核错例 | 新观察应能区分 | 必须保留的限制与反例 |
|---|---|---|
| Accrued expenses 替代 Deferred tax assets | 来源账户归属，而非只找数字 | 正确 liabilities 项不能随之被否定；来源／角色歧义为 unknown |
| annual coverage → monthly，开票仍 annually | coverage period 与 billing frequency 两个角色 | 不把 annually 误认成错的月度保障；缺 coverage 证据为 unknown |
| revenue recognition，结果17290 | 相同数值不证明账户正确，来源是 unrecognized tax benefits／gross liability | 算术对、账户错仍须分开；数字需执行时合同返回 unknown |
| 5829−5735，却说 decrease of8 | 金额、方向及比较角色 | 原 FRED 问题缺失时，不凭空假定比较顺序；silver 只标 `$8 million` 不等于 decrease 正确 |
| 46−(−34.8)=80.8 USD million | 正确派生量不能因来源没有字面80.8就判错 | 没有执行器时明确 unknown，不能宣称已验证推导 |
| 第一／第二机器人事件错归属 | 同词袋 actor／event 绑定 | 交换角色应改变答案；指代不明保留 unknown |
| over300years BC → around300years ago | 绝对年代与相对时距 | 缺参考时间不能任意换算；恢复 BC 不等于证明 ago 的换算 |
| RestaurantsReservations=False → takes reservations | 字段作用域与否定 | False→None／缺字段应 unknown，不凭餐厅风格猜测 |

同词袋角色交换、关键证据移除、False→None、账户替换、别名与缺时间锚等闭合控制，要和自然固定错例分开报告。强基线是同一冻结模型、共享自动框架的直接来源核验；统一输出上限不是严格等总算力，必须报告实际 prefill／decode token、调用和时间。若 blind 不增加可靠信息，优先保留 direct，不能靠新名字宣称贡献。

## 旧路线重叠与执行门

`source_condition` 已经从 prompt KV 重新启动局部单元；`repair_support` 已有来源×候选未来的均匀序列混合。普通 prefix reset、repair rename、rank 权重、TV／NLL／entropy、token_state 和 adaptive 混合都不能当本次新观察。proof compiler 的0支持／0冲突也包含输入合同覆盖不足：Data2txt 是 Python 风格字典，并非严格 JSON；这不等于所有 typed 检验均无效。

顺序固定为：Q0 tokenizer／接口资格 → 提议覆盖与 JSON／输出预算 → 三模式自动观察 → 同词袋绑定／缺证据控制 + 实际成本 → 选择后另行冻结完整 token 投影与 unknown 评分 → 全任务评价。提议输出上限在第一次推理前设为1536，观察上限256；达到上限必须单列 coverage limitation，即使返回恰好合法 JSON。任何实际推理还要经过根代理的资源准入与持久实验登记，保留其他用户进程，不修改 CPU `.venv`。

## 一手文献与新颖性边界

[Snowballing](https://arxiv.org/abs/2305.13534) 提供早期承诺的机制动机；这不是对原数据生成器机制的现成证明。[CoVe](https://arxiv.org/abs/2309.11495)、[QAGS](https://aclanthology.org/2020.acl-main.450/)、[QAFactEval](https://aclanthology.org/2022.naacl-main.187/) 与 [RefChecker](https://aclanthology.org/2024.emnlp-main.395/) 已有问答／独立核验／claim 检查，因此本合同不是新 QAG 贡献。

[FinQA](https://aclanthology.org/2021.emnlp-main.300v1.pdf) 与 [PAL](https://arxiv.org/abs/2211.10435) 支持显式执行与程序验证的必要性，但执行不能证明语义绑定正确。[ReDeeP](https://arxiv.org/abs/2410.11414) 与 [Lookback Lens](https://arxiv.org/abs/2407.07071) 属于需对照的来源使用研究，不能隐藏其中监督成分而替代无监督主线。[Dataset shift](https://arxiv.org/abs/1906.02530) 提醒校准不自动迁移；[Cannot Self-Correct](https://arxiv.org/abs/2310.01798) 说明反思提示本身不保证新证据。

对应机器记录：[mechanism_candidates.json](mechanism_candidates.json)。保留全部11篇文献、既有代码重叠、逐例预期、资格门和版本；没有新增性能或顶会水准结论。
