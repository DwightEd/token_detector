# 当前方法的数学含义与边界

绑定提交 `1210120748b3a05bd271bc79debcbed4fdc0c964`。这里的新命题是对本项目实现的推导；没有声称这些基础概率结论具有文献首创性。CPU 枚举只验证实现与命题的一致性。

## 1. 观测量：固定回答历史上的来源对比

记来源为 S，剩余提示为 x，已有回答为 y，定义

\[
q_{1t}(v)=q_\theta(v\mid x,S,y_{<t}),\quad
q_{0t}(v)=q_\theta(v\mid x,\operatorname{delete}(S),y_{<t}),\quad
d_t=\log q_{0t}(y_t)-\log q_{1t}(y_t).
\]

代码：`inputs.encode_source` 删除与来源字符区间有交集的整枚 token；`model.score_continuation` 使用原始回答作 teacher forcing；`observer.Observer.measure` 形成 `[N,2]` 的局部/完整历史差值。**d 越大表示该 token 相对更像来源缺失条件，并不直接表示错误。** 删除可能同时改变边界 token、位置、语法和上下文长度。保留原始回答历史的实验回答的是条件敏感性，不能冒充重新生成整条回答的总效应。

若 q0、q1 都是指定的归一化自回归分布，链式法则给出

\[
\sum_t d_t=\log\frac{q_0(y_{1:T}\mid x)}{q_1(y_{1:T}\mid x,S)}.
\]

这不需要 token 相互独立。当前实际测量中 present 使用 eager attention、absent 使用 SDPA，并保留历史 FP32 似然运算；数学身份不意味着浮点实现逐位一致。精度/后端差异必须作为工程对照。局部历史列在每个片段重置回答前缀，不能与完整回答似然混称。

这里“当前实际测量”特指默认文本 `measure → Observer.measure` 路径。`experiments.fullgap_dataset.collect` 已声明 `backend='matched_sdpa'` 并调用 `measure_responses → counterfactual.observe_responses`，两个世界都用同一个 SDPA scorer。因此**不能用默认 API 的后端差异否定全部历史 full-gap 结果**。每个历史运行是否采用该路径仍须结合其实际 plan、源码 hashes 与封存观测确认；本轮没有下载全部运行 artifacts 或重跑模型。

## 2. 默认 shared：交换型专家分配模型

一个固定片段有 n 个 token。定义隐变量 Z_i∈{0,1}，1 表示选择来源缺失专家；π~Beta(a,b)，Z_i|π 独立 Bernoulli(π)。在观测到的固定历史上使用因子 exp(Z_i d_i)：

\[
P(z\mid d)=\frac{\exp(\sum_i z_i d_i) B(a+K,b+n-K)}{B(a,b) Z(d)},\quad K=\sum_i z_i.
\]

默认 a=b=1，输出 s_i=log[P(Z_i=1|d)/P(Z_i=0|d)]。它是声明模型中的**专家分配 log odds**。实际标点分段依赖已经观测的回答；这里在给定分段上解释该条件模型，不额外宣称存在一个未指定停止规则的全局生成模型。

前向系数 F[t,k] 累加前 t 个 token 中恰有 k 个 absent 状态的未归一化权重：

\[
F[t+1,k]=F[t,k]+e^{d_t}F[t,k-1],\quad F[0,0]=1.
\]

代码在 log 空间用 logaddexp，终端乘 Beta 积分，后向消息把剩余 token 和终端积分纳入。`cavity_scores` 将 own evidence 与排除自己的 cavity odds 相加：s_i=d_i+c_i(d_{-i})。时间 O(n²)，内存 O(n²)；各片段独立处理，总复杂度为 Σ n_j²。`fit_prior` 的经验 Bayes 是可选研究工具，默认 SharedDetector 不调用它。

## 3. 命题 A：同片段排序严格保留

**假设**：共同混合先验 μ(π) 在 (0,1) 上有正质量，所有 d 有限，片段固定。无需 μ 是 Beta。

对 i≠j，令其余 token 的条件似然乘积为
G(π)=∏_{k≠i,j}[(1−π)+π exp(d_k)]。两种不一致状态的未归一化质量分别为

\[
W_{10}=e^{d_i}\int\pi(1-\pi)G(\pi)d\mu(\pi),\quad
W_{01}=e^{d_j}\int\pi(1-\pi)G(\pi)d\mu(\pi).
\]

共同积分 C>0；一致状态在边缘概率之差中抵消，因此

\[
\operatorname{sign}(P_i-P_j)=\operatorname{sign}(e^{d_i}-e^{d_j})
=\operatorname{sign}(d_i-d_j).
\]

logit 严格单调，故 sign(s_i−s_j)=sign(d_i−d_j)，包括原始 ties。**固定片段内部 AUROC 完全不变**；片段内部 AP 也不变（同一 ties 处理）。这不推出跨片段的 pooled AUROC 不变，也不推出 token 阈值、跨片段校准或区域 AP 不变。只调 a,b 无法提升同片段排序。退化在 π=0/1 上的先验需要另外处理，不能套用严格性。

## 4. 命题 B：默认模型的借用方向与自证据斜率

令 m_i=P(Z_i=1|d)。有限指数族微分给出

\[
\frac{\partial s_i}{\partial d_j}=
\frac{\operatorname{Cov}(Z_i,Z_j\mid d)}{m_i(1-m_i)}.
\]

j=i 时导数为 1，与 cavity 不含自身一致。j≠i 时，给定 π,d 的各 Z 条件独立，且
p_i(π)=πe^{d_i}/(1−π+πe^{d_i}) 对 π 单调递增。
全协方差公式将交叉协方差化为 Cov_π(p_i(π),p_j(π))≥0；最后一步可用独立副本 π′ 的差积期望证明。

所以默认交换模型不会出现“提高其他 token 的 d 反而降低本 token 后验”的负影响。这个结论依赖模型，**不能延伸到同时推断共享 persistence 的 V3**。V3 的模型证据竞争允许反向影响；旧论文已有有限反例，不应删除。

## 5. persistent / witness：精确积分解决计算，不解决语义识别

`sequence.persistent_scores`：

\[
P(Z_t=1\mid Z_{t-1},\pi,c)=c Z_{t-1}+(1-c)\pi.
\]

默认 π,c 各 Uniform(0,1)。Gauss–Legendre 对有限次数多项式精确积分：给定节点用两状态前后向递推，再对**未归一化联合质量**求和。不能把节点后验等权平均。默认每轴 Q=⌊n/2⌋+1 个节点，G=Q²，主要复杂度 O(nG)=O(n³)、前向内存 O(nG)；固定 π 或固定 c 的消融会降维。数值仍有浮点误差。

witness 把 d 替换为 w=log[(1+exp d)/2]，对应 (q1+q0)/2 相对 q1 的预测混合比。它保留了“风险区域也可生成来源可预测 token”的可能，但没有加入新的事实观察。onset、token_state、adaptive 是不同声明模型，不能把其输出均称为同一个后验。

## 6. 命题 C：仅凭这组分数不能识别事实标签

令 O 为检测器可见的概率、隐藏态、注意力或其派生量。若不规定事实支持标签 H 与 O 的关系，可以构造两个联合模型：P_A(O)=P_B(O)，但 P_A(H=1|O)=0、P_B(H=1|O)=1。任何只依赖 O 的检测器在两个模型上输出相同，因而不可能同时恢复 H 后验。即使额外阅读文本，没有外部语义规则/真实来源假设，也不自动得到分布无关保证。

该论证说明**识别假设缺失**，不是“现实中检测永远不可能”，也不是合理语义任务上所有文本信号都无效。可行桥梁包括可靠来源下的语义蕴含、合法的事实干预、独立人工标签、或明确受限的数据生成模型。Neyman–Pearson 只在指定事实类条件分布下提供最优检验；上下文对比的两个 q 不是这两个分布。

## 7. 可用但有前提的理论工具

| 工具 | 能支持的结论 | 本项目不能直接跳过的前提 |
|---|---|---|
| 似然比检验 | 指定简单假设下的检验最优性 | 必须给出事实/非事实的类条件规律 |
| Huber robust testing | 指定污染邻域的最不利似然比裁剪 | 无法从任意 clip 常数自动获得鲁棒保证 |
| 条件随机化检验 CRT | 正确条件替换下有限样本有效 p 值 | 来源替换必须来自正确条件分布；任意删除/随机文档不满足 |
| Conformal Risk Control | 交换样本、单调有界损失的边际期望风险控制 | 需要独立带标签校准；按来源/回答而非 token 交换；不是条件保证或任意 FDR 保证 |

原始资料：[Neyman–Pearson 1933](https://errorstatistics.com/wp-content/uploads/2014/08/neyman-pearson-1933_on-the-problem-of-the-most-efficient-tests-of-statistical-hypotheses.pdf)、[Huber robust estimation](https://ir.cwi.nl/pub/13042/13042D.pdf)、[Model-X CRT supplement §F.1](https://faculty.marshall.usc.edu/jinchi-lv/publications/JRSSB-CFJL18_Supplementary.pdf)、[Conformal Risk Control](https://arxiv.org/abs/2208.02814)、[Pearl 2009](https://escholarship.org/content/qt36w8n7pg/qt36w8n7pg.pdf)。它们支持各自条件下的理论，不是本项目已有事实检测保证。

## 8. 精确分解 pooled AUC：哪些比较在驱动收益

正类标签在本项目应定义为**不受来源支持**；外部事实错误是另一目标。设所有已知正负 token 对的集合为 P，固定片段内的对子集为 W，跨片段子集为 B。每对比较 C(s+,s−)=1{s+>s−}+.5·1{s+=s−}，则 P=W⊔B，因此

\[
 AUC_{pool}=\frac{|W|AUC_W+|B|AUC_B}{|W|+|B|}.
\]

证明只是把全部对的 concordance 总和按互斥集合拆开；分母相同。若某子集零对，其条件 AUC 未定义，报告 None 而不是0。AP 不具有这里的线性身份。该基础分解本身不被包装成新统计理论。

`experiments.local_evaluation.auc_pair_decomposition` 用负类排序与左右 searchsorted，整数保存两倍 concordance（胜=2、平=1、负=0）；不显式构造全部 O(N²) 对，运行 O(N log N)、内存 O(N)。已有 fullgap 评价在分数解封后、unknown 标签排除后调用，附带在每个方法的指标中。它不更改分数、读取顺序或旧输出。不同回答/片段范围按原连续轴隔离，纯标签片段只贡献跨段对。

这里 AUC_W 按 token 对数量加权；它与“每来源等权、每回答/片段宏平均”的科学目标不同。当前 `paired_span_differences` 的点估计是 pair-weighted，bootstrap 单位是 source。不能把“按来源重采样”说成“按来源等权”。未来主实验若选 macro/source 等权，须事前冻结并实现另一显式估计量，不能看结果后切换。

## 9. proof_revision 已实现的有限语法与 KL 投影

`proofs.py` 不是开放语义parser。它只识别嵌入JSON年份行表、明确单位、规定格式的直接/加减断言；表不完整、key缺失/矛盾、单位/作用域/字符投影不明确或枚举超预算，一律unknown。不兼容断言编译为“至少修改这些角色之一组”的单调DNF，**支持断言不生成clean标签**。role可以是subject、year、数字幅度、显式符号、操作符、单位；没有文本位置的隐式符号不造token。

`revision.py` 把每个角色是否至少有一个latent state=1编码成OR bit；同一修复项的角色AND，不同项OR，不同proof因子AND。块内在V3状态轴之外增加2^R的mask轴，块间通过OR卷积连接；默认R≤8，否则保持基础分数。这个事件是声明模型中的修复兼容事件，**不等于角色的真实语义错误已经被识别**。

记基础状态分布P、事件A的质量p=P(A)>0，允许分布Q满足D_KL(Q||P)≤B，并最大化Q(A)=a。KL的分区链式分解给出

\[
D_{KL}(Q\Vert P)=D_{KL}(\operatorname{Bern}(a)\Vert\operatorname{Bern}(p))
+aD_{KL}(Q(.|A)\Vert P(.|A))
+(1-a)D_{KL}(Q(.|A^c)\Vert P(.|A^c)).
\]

在给定a时，保持两个分区的内部条件律即可达到最小KL，故只需在a∈[p,1]求最大的二元KL可行点。若−log p≤B，a=1；否则用brentq求KL(a||p)=B。代码用有限log p而非先舍掉罕见事件，预算默认log2。

令w=(a−p)/(1−p)，最小KL修改可写作Q=(1−w)P+wP(.|A)。所以token边缘概率也按同一w混合原V3概率与hard条件概率。各分区内部条件律保持，修改量受a−p限制；**预算约束不提供proof正确率、检测准确率或概率校准保证**。这是一项已有实现的变分包装解释，并未被当作新论文的首创贡献。
