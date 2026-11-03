# H012：自然 NER 槽的擦值/来源候选库存，不是可用 QA

2026-11-03。状态：设计已由 root 审核最小合同；本阶段只实现纯 deterministic compiler 与无标签 artifact/成本库存。不得调用 QA、GPU、生成模型、tokenizer、spaCy 或新 NER，不读取 evaluation_only/gold/predictions，不增加模板、模型或参数搜索。root 负责独立 CLI、完整输入冻结、installed release 与本地实际运行；本文件不宣称运行结果。

## 已知负结论必须保持

H008 严格 toy 的 source-defined Q 资格通过，不等于自然 R→Q。root 已报告封存 H011 的固定 direct masked-cloze 在 0/4/12 负担分别只答对 9/16、8/16、10/16，最短条件已经失败；不能把失败单独归因为后缀，也不能把已可用的 7B 变成 prompt/模型搜优。H012 不给 H011 换一个更强提问指令，不把自然擦值编译成功写成 QA 能力恢复。

因此 H012 只产出 cloze artifact、source-only candidate inventory、信息流诊断和潜在成本。**不生成 question/system/messages，不准备模型执行请求，不声称其中任何题面语义唯一可答。** 原问题、锚、别名、候选域、算术闭包和断言状态均未被该编译器解决。

## 固定材料与计数

只读取已冻结的：

- H010 的 pilot-inventory.jsonl、pilot-summary.json、pilot-seal.json 和模型/工程身份 metadata。
- 原自然 18 答的 inference-only inputs.jsonl，仅使用 id/source/response。

全部 18 原答、H010 已输出的 719 个 source entity occurrence 与 136 个 response entity occurrence 保留。复用 spaCy 3.8.18/en_core_web_sm 3.8.0 外部监督 NER/POS/dependency 结果；不重新运行其管线，不按任务、类型、原值是否出现在 S 或任何后验效果筛选目标。136 是已输出的 occurrence 数，不是语义覆盖、所有实体数量或事实可答率。

outer stage 必须先验证 H010 seal/固定文件 inventory 与 source/response 文本 SHA、原 18 id/顺序、Unicode 字符长度和 719/136 完整计数。纯 kernel 对小合成样本可泛化，不硬编码 18/136，但生产 profile 不得减少固定总体。预算失败或异常不丢原答。

## 纯接口与输出

新增唯一纯模块 experiments/research/natural_masked_questions.py：

    prepare_natural_questions(samples, inventories)
        -> {schema, originals, source_pools, proposals, counts, limitations}

samples 提供 id/source/response；inventories 使用 H010 原字段结构，每个 id 有 source/response 两个 field inventory。kernel 只依赖标准库字符串/哈希/JSON，不导入 spaCy、torch、transformers、模型、数据文件或 scorer，不做文件 I/O。

originals 按原输入顺序保留所有 id、文本身份和全部 proposal id；proposal 按原 response.entities occurrence 顺序保留，不按唯一值去重或只留“可问”的 span。每个 proposal 显式分离：

- compiler_status / compiler_reason：是否仅完成安全、可复核的字符串编译。
- cloze_artifact：masked_response、[TARGET] 占位符与原生 NER label；没有 QA 指令、问题文本或 messages。
- target_readout：原 occurrence id、原 exact span/text、原 label，仅供库存/对齐；不能回流到 cloze_artifact。
- mask_spans：原坐标/原文到 placeholder 的完整映射、字面重复计数。
- source_pool_id：指向先由 S 建好的同原生 label 候选池。
- semantic_status：anchor、candidate-domain completeness、alias、arithmetic、assertion/factuality 均为 unknown_not_assessed；qa_admitted=False。

没有 risk=0、entailment、confidence、正确/错误或任何预测分数。原答未提案区域、NER 未检出的实体与全体历史 token 都仍为 unknown/unscored，不通过空数组或默认 0 伪装完整检测覆盖。此阶段不新建 token 轴，亦不把 NER tokenizer 计数冒称 Qwen/BPE token 成本。

## 来源池在处理 R 值之前构建

对每条 source field 的原生 entities，按原 type label 独立分组。每个 label 的候选是原 exact text 的规范字符串排序；只按完整 raw text 完全相同去重，不能 lower/strip/Unicode normalize/合并别名或按 R 值优先。source occurrence 原 id、start_char/end_char、text、type 和 owner id 都保留；重复字符串的每次 source occurrence provenance 不丢失。

目标的原生 label 仅选择已经存在的同 label source pool，不修改其成员或排序。其他 label 不自动合并。空池、单候选、多个候选分别记录实际 cardinality；它们的语义完整性一律 unknown_not_assessed，单候选绝不是置信度 1。

source 字段哈希/结构/offset 不可信时显式保留 source-inventory-unknown，不从部分错误 inventory 猜池。来源实体在 S 出现只是候选 provenance，不是问题可答或候选事实正确的证明。来源缺少某 R 原值、NER 未认出别名或合法推导值不在表面池中，都不能直接判 R 错。

## 固定擦值规则

每个目标 occurrence 必须由已冻结 H010 occurrence id 标识，原 text 必须与 R[start_char:end_char] 完全一致，坐标单位是 Python Unicode code points 的半开区间。不得通过搜索“修好”错误 offset。

1. 在整个原 R 中枚举目标 raw text 的所有字面 occurrence，含不被 NER 标出的重复，含重叠匹配；全部应被擦除为 [TARGET]。
2. 同时纳入 R 中所有同原生 label 的 NER occurrence；目标字面值覆盖的同坐标 span 用 [TARGET]，其余按原坐标顺序使用 [OTHER_0001]、[OTHER_0002] 等固定 occurrence placeholder。
3. 同坐标、同边界的重复 mask 项只合并记录其 provenance，不重复替换。其余 mask 区间发生重叠、边界冲突、越界、原文不匹配、空 literal，或原 R 已含预留 placeholder 时，该目标 compiler_status=unknown；不选较长 span、不合并语义 label、不修字符。
4. 从原坐标一次性切片拼接；未遮蔽的字节对应 Unicode 字符序列/标点/空白/换行/后缀保持不变。验证原 R 中每个目标 literal occurrence 都已被替换，没有来自原 R 的未擦除目标 occurrence，全部实际替换映射可重建。固定 placeholder 常量内偶然包含同一子串（例如目标是 1，而固定标记是 [OTHER_0001]）不携带随目标值改变的复制信息；另记常量子串诊断，不把这种常量巧合当作原值泄漏或改 marker 的理由。

不同 [OTHER_n] 只是 occurrence 标记，不声明它们一定是不同实体；相同 literal 的多个 [TARGET] 也不是经过验证的共指。**没有 H011 的“另一个人一定不同”、双射、恰好两人/两事件或正确 event 锚假设。** 其他类型上下文可能提供必要条件，也可能错误/泄漏；仅保留交叉/上下文诊断和语义 unknown，不能称原值擦除等于语义无泄漏。

placeholder 方案只有这一种，不根据原案例效果换 marker、扩大/缩小类型集合、改句子范围或增加关系词规则。

## 诊断与成本，不向前馈送答案

池在先，mask 在后，最后才允许计算 original_value_exactly_in_source_pool 作为只读诊断。这个布尔值既不改变候选，也不改变 proposal 是否保留；不能被当作支持/矛盾、QA admissibility 或 risk。它也不构成别名等价判断。

库存完整报告 18 答/136 proposal 的 compiler 状态与原因、原生 label 分布、字面重复/同类型 mask 数、source pool 的 raw occurrence 与 unique-value cardinality、masked character length、需要另审的输入范围。潜在 completion pair 数只能标作“若未来评分”的字符串库存量，不是本阶段模型调用，也不凭字符/NER token 数估算真实 Qwen 内存或生成成本；实际模型/tokenizer/QA 调用数固定 0。

即便一个题面字符串编译成功，它的锚、域完整、alias、合法算术、非断言/否定作用域、可答性仍全部未知。不得把 compile success count、候选条数或新增规则数量写成语义覆盖，更不能写 AUROC/精度改善。

## 对照历史与唯一下一研究方向

H002 是自由字符坐标提议前端，核心来源 QA 当时 0 调用；H008 用已知 toy 语法和 source-defined Q；H011 用已知 toy person 槽与双洞 R-cloze，而且实际迁移失败。H012 只复用已冻结自然 NER occurrence，解决其原文坐标/重复/类型池/擦值 artifact 合同，不重演自由 offset 生成，不声称修复 H011，也不是把旧有限证书的零覆盖换名成检测器。

下一研究问题最多保留一项：**S 独立定义的、可核查锚的 query universe，如何与 R 的自动关系表示匹配，并对多种支持修复保留歧义。** 这还不是本阶段实现内容。原 Q 若无可靠恢复不假设存在；本次 FRED 固定 raw-doc0 回连恢复 0，不能据“上游有 question 列”强行锚定。

来自 S 的 QA/从 R 提取细粒度 claim 均有 QAGS、FEQA、Q²、QuestEval、QAFactEval、RefChecker 等先例；必须另核 prior31 真实执行状态和新增信息。source-defined universe 不能只挑 S 中易答事实后忽略 R 中不匹配/未覆盖的断言。H012 的池/擦值库存不解决这个问题，不默认给未匹配 R 低风险，也不启动新 GPU 实验。

root 提出的“已知 toy 单谓词/单洞”对照另待 prior/P026/sentence-scope 审查；本阶段不偷渡为第二模板或自然模型观察。不因 H011 负结果更换 prompt 或调用更大模型救分。

## 验证与停走

测试只用小合成字符串/H010-shaped 假 inventory：候选对 R 值/次序变化不变、source 重复 occurrence 保留、全 literal 重复擦除、同 label 全擦除且其他类型原样保留、Unicode/emoji/组合字符坐标、同坐标合并与部分重叠 unknown、placeholder collision、空/单/多池、原 case/occurrence 顺序和 no-input-mutation。测试不验证语义正确性，不跑真实 NER/模型。

outer audit 在打开 input/inventory 值之前固定只读允许路径、原 seal、source/model/script/协议与完整 case/occurrence 预期；输出写入独立新目录并封存，异常保留不自动覆盖。若有 target 原值泄漏到 cloze artifact、R 值改动 source pool、原 occurrence 消失、未提案 token 获得假 0 风险或语义 unknown 被写成可答，即拒绝该 artifact 合同；不依据 benchmark 标签改规则。

完成只意味着库存可审，不意味着获得可运行/有效自然 QA。自然语义锚和完整候选仍是未完成的研究问题。
