# H014：H011 缓存矩阵的固定双射读出，零新增前向的必要控制

2026-11-03。状态：**只写前瞻合同，尚未计算 H014 结果；本审未读取实际 raw LL/参考来选方案，也未运行模型。** 在 H013 新观察前固定这一必要缓存控制；H012 仍只是自然 artifact 库存。H011 的独立行失败完整保留，不由结构化结果覆盖或事后挑赢家。

这是已暴露 H011 上的 posthoc cached diagnostic，不是新的未接触确认集。动机来自 root 已封存的同名双角色 collapse 审计；数值读出则在 H008 结果之前的 STRUCTURED_FOLLOWUP 及 H009 中已固定并实现。本轮不能把“kernel 已预先固定”写成“对 H011 的这项应用在看到其结果前已经确认”。

## 唯一问题与竞争解释

对显式 toy 双射，独立两行都选同一个来源名字，可能仍保留正确的跨角色对比分量；也可能对比分量本身就指向错误关系。用相同四个 source-QA 完整姓名+EOS 原 joint LL、相同冻结两可行指派读出，区分这两类观察。**不改变 prompt、模型、候选、EOS、原 H011 score、温度、长度、offset、阈值或任何权重；没有新增 nuisance arm。**

如 joint 正确而独立行 collapse，只能说该显式可行域下的固定结构化读出利用了四格的相对信息。它不唯一证明 name-frequency/长度偏好导致原错，也不证明 LLM 理解了自然角色。任意 2×2 表都可作加性主项/交互分解；消去加性列项是代数性质，不能由一个读出改善反推出真实生成机制。

若 Delta 错误、tie 或 unknown，保留失败，不重新调 offsets 或换 normalization。若 H013 单 conjunct 改善而 H014 仍弱，说明增加显式双射不足以替代题面 bundle；若 H014 改善而 H013 仍弱，也不能因而选择 joint 为“新主方法”。两者都是同一已知 toy 的必要诊断，不是六任务效能结论。

## 数据与唯一分组

唯一神经来源是 H011 已封存的全部 96 completion 观测：48 unique masked Q × 两个来源人名。按 source exact identity 与已冻结的 0/4/12 后缀构造分组，得到 **8 sources × 3 burdens = 24 个矩阵**。每矩阵两行按完整 event 字符串规范顺序、两列按完整来源人名规范顺序；两行恰是该 source/burden 的两个目标角色问题。分组必须来自 inference/compiler 的来源、event 和原 suffix bytes，不使用支持/错误标签、原 R 值或 raw LL 排序决定行列。

所有原 48 回答/96 槽、192 logical completion 映射完整保留。两种原 R direction 对同一矩阵共享观测；不当作两份独立 QA。24 矩阵也不代表 24 个独立新来源：只有八个来源，三种已暴露负担条件。源端候选固定为两个人名；alias、域外值、缺人、额外人/事件或未满足唯一关系，均不得因矩阵有 2×2 shape 就准入。

source_is_admitted_bijection=True 必须来自 H007/H011 已冻结、只读 source 语法的显式两事件/两人一对一事实合同，不能从两行的模型输出推断。该 parser 自身已经能解 toy；因此本控制不能证明模型必要性。

## 数学与代码完全复用

唯一数值实现是 experiments/research/bijection_readout.py 的 assignment_distribution；contract=two-by-two-admitted-bijection-v1。冻结当前源码身份，不复制另一 scorer，也不改 H009 kernel：

\[
L_0=\ell_{00}+\ell_{11},\qquad L_1=\ell_{01}+\ell_{10},\qquad
\Delta=L_0-L_1,\qquad
\widetilde p_k=\frac{\exp L_k}{\exp L_0+\exp L_1}.
\]

ell 是每个完整候选名所有 token 与固定 EOS 的原 full-vocabulary-normalized joint log probability；不能用两候选 softmax probability、risk 或新拟合标量替换。两指派仅为 [0,1]/[1,0]。math.fsum、exp-shift、精确 computed-energy tie、finite/非 bool/overflow→unknown、marginals 和 underflow 记录按原 kernel 原样执行；尤其不能因浮点概率同为 0.5 而抹去非 tie 的能量 MAP。

两能量之和不假设两个事实条件独立，归一化质量是条件于显式双射可行域的 energy mass，不是校准事实后验。行/列加性项同时给两能量加常数；有限精度容差只用于报告数值一致性，不作为按标签选择的 tie 阈值。

保留两个固定并列读出：

1. **independent rows**：严格重建 H011 的每行两候选条件概率、greedy/tie 与原 max-slot case risk，原 source/event flip 和全部成本/unknown/失败原样保留。
2. **joint assignment**：原 R 两槽按 canonical event 顺序对应一个合法指派时，case risk 固定为 1−对应 assignment mass；不合法 R mapping 或 kernel unknown 保留 unknown/risk1，不能投影成“最接近”合法指派。query marginal argmax/tie 从精确能量 MAP/tie 导出；marginal 概率作为数值结果另存。

没有第三臂、合并权重、按 case 取优、任务赢家拼接或用标签校正符号。

## 冻结、释放与完整性

1. 在任何 payload hash/重建前，先核完整 H011 archive 路径/别名/根与 child inventory/seal keys；保留 initial root seal。
2. 使用明确兼容的 H011 已冻结 observer/helper/profile/模型身份，重建全部 input/compiler/48Q/96 requests、token metadata、原 raw logits/LSE/logp/joint 与成本；不调用 scorer 或 model forward。若需要 tokenizer 验证，仅离线 fast tokenizer，不进入 GPU _prepare/_ensure_model。
3. 先构建全部 24 原 raw 矩阵和两固定 readout、所有 48 case 映射，保存并冻结 CPU plan/protocol/source/score artifacts。只有此后才打开 H011 的已封存 CPU review 参考。
4. 参考必须绑定同一 initial donor seal、原 input/source/response SHA、完整 IDs/槽位和 compatible helper bytes。原 H011 summary/queries/cases/raw/costs 均应完整重建比对，不能只抄 favourable aggregate。
5. 输出使用精确 staged inventory、固定首次 payload hash map 与 readback equality；最终 root seal 不得为后改写的文件重新背书。模型资产、初始 donor/reference seals、CPU freeze 和全文件 inventory 在释放/导出前后持续复核。失败撤销 complete/seal，保留失败材料，不自动重试或重算观察。

H011 的原参考已暴露这一事实不会被上述新的 CPU freeze 隐藏；冻结门的作用是禁止本次计算后再调规则。

## 固定报告与判别

主报告是全部 24 source×burden matrix 的完整表：canonical row/column identity、四个 raw LL、两能量、Delta、MAP/tie/unknown、conditional masses/marginals，以及释放后固定来源参考指派。每个负担分别报八个 source assignments 的 independent-both-roles 与 joint 正确/wrong/tie/unknown，及 16 个 query 的独立行/联合边缘 top1。保留全部 discordance cells：independent collapse 且 joint 对/错/tie，独立已对但 joint 损坏，以及双方失败。

次级完整 48 case 风险、AUROC/AP、所有原槽位和 unknown 另存。不同 readout 的概率空间不同；AUC 排名比较不等于置信标定比较。H011 原 source/event-flip 全指标独立保留；任何新 derived-marginal 指标须标作 CPU-derived，不称原始 QA 神经观测。没有新的语言观测、token 轴或自然数据指标。

成功只支持“该 toy 条件下存在固定 joint readout 可利用的 residual role contrast”；不以恢复多少例为由改门槛或晋升。失败支持“相同已采 QA 矩阵在该可行域下仍不足”，不是整类约束推断不可能。无论结果均与 H013/H011 原结果并列保留，不能挑较佳臂宣布主线提升。

## prior 对照与预算

| 先前项 | 为什么保留，以及本轮是否重复 |
|---|---|
| 观察前 STRUCTURED_FOLLOWUP / H009 | 两合法指派能量、Delta、加性行列不变性与高置信错误反例均已定义；H014复用同一算法。本轮新材料是 H011 R-masked、含负担且已出现 collapse 的矩阵，不再运行 H009 baseline/stress，也不换 offsets 造改善。 |
| H008 / H011 | 前者 source-defined Q 后者 R-derived compound Q；全部原观测保持。H014只在后者缓存上加先前固定的显式可行域读出，0 新前向，不声称新的模型信息。 |
| P026 / P021 | 原 R future repair/source worlds 的 teacher-forcing，与这里固定 QA 矩阵条件读出不同；不重放旧 future、null source、year repair 或 prefix-reset。 |
| P022–P025/P027 | 约束/能量/有限证书思想有先例；assignment 不是新算法，换能量名称不能建立新颖性。本控制必须用已有固定 kernel，不重新拟合这些旧核或将语法匹配叫真值。 |
| H013 | 已知单 conjunct 的新消息 bundle，是另一独立的必要能力干预；H014作为其零 GPU 缓存控制，在其新观察前固定，不覆盖任何 H011 失败。 |

执行预算：**new model/QA forward=0**，无新模型/大下载、无 nuisance 参数或 prompt sweep。只读现有已封存缓存与所需离线身份资产；CPU 时间实际记录，不能把原 H011 96 次前向成本算为 H014 新成本。当前只是设计，未执行计算、未获得效果结果。
