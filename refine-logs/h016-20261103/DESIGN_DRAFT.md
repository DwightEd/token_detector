# H016 草案：同一 shared-prefix 仪器下的 compound / atomic 配对诊断

2026-11-03。**DRAFT，未登记运行、未准入模型、未产生新分数。** 本文仅固定待审核的问题与边界；只有 H015 的两棵 synthetic trie 工程资格实际通过、向量持久化和同源纯 kernel CPU replay 封存后，才可另行冻结 H016 的完整 token-only manifest、预算和观察协议。当前不启动新的 primary module，不读取新 raw、预测或评价参考。

## 唯一问题

在同一合法 shared-prefix 概率树仪器上，分别保留 compound 与 atomic 两种题面，再对每种题面并列使用独立行和固定双射联合读出。问题是：去掉非目标 conjunct 的题面组合变化，与显式合法联合读出，是否在此已知 toy 域中提供不同的可恢复关系信息？

这是一个已暴露构造域上的配对诊断，不是自然检测器或未接触确认集。H011、H013 的旧观测合同不同，不能直接以旧 compound 分数配新 atomic 分数作这一主比较。H013 原严格概率门失败永久保留；新仪器上的另一次配对观察不为旧 32 次前向补写 success。

## 固定人口与两种题面

- 主人口只取 H007 原 inference 的 burden0 全部 16 原回答，底层为 8 个来源、两个明确事件角色。保留原 supported/swapped 方向全部实例，不能按先前结果选择来源或槽。
- 每个 scope 有 8 source × 2 role = 16 个唯一问题，每问题两个完整 source 姓名候选，共 32 个完整 candidate+EOS 路径。两种 scope 合计 32 个问题、64 条候选路径；这不是实际 forward 数。
- 每个 scope 的 16 原回答 × 2 槽产生 32 个逻辑问题实例、64 个逻辑候选实例，均保留指向唯一问题/路径的映射。两个 R direction 擦值后的重复不得当成独立神经观测。
- 来源全文、source-only 排序的两个人名候选、模型九资产、tokenizer、system、QUESTION_TEMPLATE、完整姓名加一个 terminal EOS、量化和数值 runtime 均相同。原 R 仅参与固定槽位/conjunct 编译、literal erasure 与原实例 readout 对齐；不得从 R 选择候选池或把正确绑定写进题面。

H011/H013 的固定模板逐字保留：

    Using only the source, give the full person name for [TARGET] in the masked statement. [OTHER] denotes a different source person.
    Masked statement:
    {masked_response}

system 必须从冻结 H011/H013 消息身份逐字复用，并在 manifest 中保存完整字节；不能凭本草案重写近义指令。

**compound**：复用 H011 burden0 的完整首个复合句。目标人名的所有规定 literal occurrence 擦为 [TARGET]，另一人名擦为 [OTHER]；保留原事件、连词、标点与空白。

**atomic**：复用 H013 的严格 known-slot compiler，取目标对应的一个完整肯定 conjunct 的原 R raw slice，只将该目标 literal 擦为 [TARGET]。第一 conjunct 补固定 terminal “. ”；第二 conjunct 保留原 terminal。完整模板中解释 [OTHER] 的句子仍保留，即明确的 dangling-instruction 条件。

atomic 同时去掉另一 PERSON、另一个事件谓词/洞及不同人耦合，缩短 prefix，并有第一 conjunct 的确定 terminal 规范化。允许的作用对象是这个完整题面 bundle；不能唯一归因为 attention、谓词数量、单个角色绑定或原生成器的内部因果机制。主比较不用 burden4/12，避免将删 suffix 当作新增效果。如外层保留全 48 回答/96 槽的历史映射，其 burden 坍缩只属编译复用，不产生新的 suffix 实证不变性结论。

## 一个观察仪器，四个固定读出

两种 scope 一律使用 H015 通过资格的 shared-prefix backend：每个完整候选路径包含全部姓名 token 和固定 tokenizer EOS；对每个唯一 causal context 只取一次原生完整词表 FP16 向量，使用同一行计算该节点所有出边的 float64 全词表 normalizer。prefix 内部合法 chat EOS 不等于另行追加 completion terminal EOS。

原 H011/H013 的 P+K 输入、末尾多行提取和变长 separate-forward 合同，不是这一 prefix-only、每节点一行的合同。不能把其选定 logits 或 LL 拼成新的概率树；也不能只给 atomic 换仪器。旧分数完整保留作历史背景，本轮主比较仅使用统一新仪器得到的两种 scope。消息相同不使两个不同数值仪器等价；仪器不同也不构成方法新颖性。

冻结如下四臂，不增加择优或 nuisance arm：

| scope | independent rows | joint assignment |
|---|---|---|
| compound-b0 | 完整两候选条件质量；原 max-slot risk | 原 H009 两合法指派能量 |
| atomic-b0 | 同一独立行公式 | 同一原 H009 kernel |

独立行完全复用既有 candidate_distribution 的固定两候选条件读出，原回答风险为 max-slot risk。联合读出完全复用 experiments/research/bijection_readout.py 的 assignment_distribution，不复制或修改数值核：

\[
E_0=\ell_{00}+\ell_{11},\qquad E_1=\ell_{01}+\ell_{10},\qquad
\Delta=E_0-E_1,\qquad
\widetilde p_k=\frac{\exp E_k}{\exp E_0+\exp E_1}.
\]

四格是完整 name+EOS 的新仪器 joint LL，不是 risk、截短姓名或两候选归一化后的概率。仅当原 R 映射属于显式两个合法指派时，joint case risk 为一减相应指派质量。非法映射、kernel unknown、非 finite、overflow 均完整保留 unknown/risk1，不投影到较近合法指派。精确能量 tie、稳定归一化、边缘质量和 underflow 依原 kernel 合同处理；不按标签设置 tie 容差。

source_is_admitted_bijection=True 只能由冻结 toy source grammar 的显式两事件/两人一对一 licence 给出，不由 2×2 shape 或模型输出推断。联合质量条件于这个可行域，两个 LL 相加是 energy 构造，不证明事实条件独立；它不是校准 truth posterior。source parser 自身能直接回答此 toy，故四臂任何结果均不能证明 LLM 必要。

## 模型前冻结、去重与实际预算

本轮不猜 node 数、token 数或峰值预算。tokenizer-only preparation 必须在任何权重加载前给出并冻结：

1. 原 inference SHA、全部原 IDs/S/R 字节与 scope/slot/raw-slice 映射；全部完整 messages、候选 literal、prefix/candidate/EOS token metadata、无静默 strip 或重分词的 prefix alignment。
2. 两 scope 全部 canonical trie、每个完整 context token ID、节点/边/叶映射、跨逻辑实例及跨 scope 的 exact identity 去重表。共享 causal context 的键含完整历史和 instrument identity，不能只用末 token 或 protocol hash。
3. 每个 scope 与合并后的 unique node 数、实际语言 forward 上限、逐节点及合计 input tokens、最大 context、完整 V 维 FP16 向量与 Python 临时向量存储、预计硬盘上界。64 条完整候选路径绝不冒充 64 次 forward。
4. 模型九资产、tokenizer/config、精确 runtime/quantization/attention、纯核/backend/collector/reader/helper 源码、实际 immutable installed target/wheel 的实测文件清单与哈希。不得假定 member 数是 126 或其他历史数字。
5. 全四臂参数、CPU reduction 与评价计划、完整 staged inventory、初始 seal 保存规则、资源门限、一次尝试和 failure policy。

H015 receipt 仅覆盖其实际 synthetic contexts，最大 512；4096 只是 backend rejection bound。应先报告 H016 实测最大 context：若超出已获工程资格的范围，不得以“上限 4096”冒充已验证资格，须另行明确处理后再准入。保持原资源门，不以等待或旧失败为由降低。

执行前对所有既有缓存做 exact 请求审计：只有完整 context、模型/数值仪器、完整原向量与有效 seal 均相同的已资格节点才可复用。协议或代码哈希变化本身不准许重算；旧 separate-forward LL 本身也不准许伪装新节点向量。保留所有 duplicate instance 的 provenance。新请求数必须是冻结 manifest 的实际差集，不能以“同模型”或“已换仪器”为由省略核验。

## 固定观察、评价与判别

模型阶段保留所有节点原向量 NPZ、context、出边、完整候选路径、raw logit/normalizer/logp/joint、pool mass/underflow/unknown、逐节点成本和 current/peak allocator counters。无 generated text、额外语义 warmup、sampling、prompt/model/温度/长度/阈值搜索。发生严格概率、telemetry、完整性或预算失败即保留 partial/incomplete，不 clamp、不改 epsilon、不自动重试，也不删除较差 scope。

冻结全两 scope 的原始分数、四个 readout、全部 case 映射、CPU source/protocol/analysis plan 后，才允许读取/hash 构造参考及旧 CPU review 参考。先核路径/alias/inventory 与初始 seal，再解封；最终 release/export 维持同一 initial seal，不以重封为后改写背书。该顺序防止本轮观察后改规则，不能使已暴露 toy 重新变成 held-out。

主报告完整列出每 scope 的 8 个来源、16 个唯一 QA 的两候选 LL、四格矩阵、Delta、independent/joint MAP、tie/unknown、质量与参考映射。逐四臂报告：

- 八个来源的 both-roles correct/wrong/tie/unknown，以及 16 个角色问题的正确/wrong/tie/unknown；
- 全部 16 原回答的固定风险与次级构造 AUROC/AP，不把风险质量当校准事实概率；
- source 配对 discordance 全表：independent 同名 collapse、joint 恢复/损坏、双方失败，以及 atomic 相对 compound 的所有改变；
- 同一 scope 内的 joint−independent 与同一 readout 内的 atomic−compound 描述性对比，完整报告四臂及交互，不按结果挑主臂或换聚合。

只有八个 source；两个角色和原 R direction 不是独立样本。没有预设成功阈值或显著性选择。若 scope 改变主要出现在独立行而联合两臂相近，只支持这些题面/读出上的条件差异；若 joint 仍错，说明该四格关系对比在当前仪器下不足。即便 atomic joint 最好，也不能据此唯一识别内部 attention、消除所有信息歧义或晋升自然方法。两 scope 都弱或存在工程 unknown 同样是完整结论。

## 与旧实验的边界

| prior | 本轮真实区别与禁止的重放 |
|---|---|
| H013 原协议/缓存 | 原协议固定 H011b0 主对照、相同模板和 dangling OTHER。旧 32 次 separate-forward 因严格概率合同失败，无 root success；此失败不修补。本轮新仪器同时重测两 scope，旧概率不能混用。 |
| H011 / H014 | H011 compound 接口及其负结果保留；H014 是旧缓存上原 H009 读出，0 新前向。本轮复用读出算法，不把旧 joint 改善当新仪器结果，不选择较佳历史臂。 |
| H008 / H009 | source-defined Q 的必要能力与固定 assignment 算法已有；不重复请求作新背景控制，不声称 QA 或 2×2 assignment 新颖。 |
| H005/H007 / context stripping | 标点 sentence 仍包含两个谓词；旧 A/B/C verdict 保留完整 R 上下文。删除 suffix 回到 b0 的 exact 请求复用证明仍有效，不再跑 suffix 去除“新实验”。 |
| P026 / P021 | year repair、原 R future continuation、source worlds 与 local reset 已做；本轮完整来源姓名问答不是这些 observable 的换名，不重放其 rejected readout。 |
| P003 / P002 | 删源 token 的 dependency 与 hidden residual/distance 不是真值资格；本轮不用影响/attention/cosine 替代关系正确性，不以换 divergence 或读出名称重开旧假设。 |
| H012 / source-only 900 licence | 自然 NER artifact 与 source path 功能性只给编译/来源结构库存。它们没有授权 R 全覆盖、候选完备、自然双射或自然 QA；不能将本 toy 前提迁过去。 |

## 后续自然迁移尚欠的资格

本轮没有测试 relation-preserving 词序控制、实体/事件词汇重命名、关系缺失与显式否定、null/非断言、合法 many-to-one、时间更新、alias/coreference、算术 closure 或候选域完整性。它们是后续提出自然关系模块前必须分别审查的条件；不能凭已知 toy 分数宣称已过，也不在本轮暗加一套对照或 prompt sweep。

source-only parser 可直接给字段答案的域，难点在自然 R 的可核谓词/角色对齐、别名和作用域，而非让 LLM 重述字典。H016 最多建立同一仪器下两类固定题面与两类读出的可比诊断；自然六任务目标仍须另有独立且完整的协议，不从 toy AUC 外推。

## 本次审查意见与待决项

**原则上可登记为 H015 过关后的必要配对诊断；现在不能启动。** 其可判别性来自同一仪器中的四臂完整对照，而非重新包装 H013 未成功 run。科学主张必须限定为题面 bundle 与固定可行域读出的条件差异。

待决项只有工程冻结事实：H015 sealed pass、实际 H016 node/token/storage/max-context 预算、qualified-cache 差集、模型资产/runtime/installed source identity，以及独立 review 后的完整 observation/evaluation protocol。它们不需要根据标签或旧预测选择。本文不写 core/state，不访问新 raw/refs/current predictions，没有任何模型调用。

依据：[H011 冻结设计](../../refine-logs/h011-20261103/DESIGN.md)、[H013 冻结设计](../../refine-logs/h013-20261103/DESIGN.md)、[H013 prior 审查](../h013-atomic-20261103/prior-review.md)、[H014 冻结设计](../../refine-logs/h014-20261103/DESIGN.md)、[H015 工程设计](../h015-prefix-trie-20261103/ENGINEERING_DESIGN.md)。另只读核对 H013 原输出 protocol.json；没有读取该输出的 raw 或参考。
