# V3 / witness：当前保留的研究方法

**当前要研究和解释的是 V3，不是旧 R4。** V3 在旧 Llama 金融两组有正向证据，后续替代候选尚未胜出；公开接口默认仍为 V1 / shared。二者是研究状态与软件默认值的区别。完整数字见[六组结果及所有已恢复主对照](../RESULTS_CURRENT.md)。

最新用户指示允许多模块组合，优化不再限定于V3或统一模型。新恢复结果显示R4在HalluRAG含答案组更强，见[完整对照、错例与0.8目标](PERFORMANCE_REVIEW.md)。V3保留为基线和可复用模块，没有宣称新融合已有效。

## R4 → V1 → V2 → V3，各自改变了什么

| 版本 | 观测与模型 | 带来的能力 | 尚未解决的限制 |
|---|---|---|---|
| R4 | 按任务选 local/full 来源对比，参考分位数、局部创新、route、链消息 | 多观测及任务配方；QA / Data2txt 旧结果仍强 | 额外观测/参考成本，金融迁移差；难以单独归因各项 |
| V1 / shared | 只用 full gap；片段共享 Beta(1,1) 专家倾向 | 无 FIT 的统一潜变量模型，保留自身证据 | 固定片段内严格保留 raw gap 排序，不能改善该粒度局部 AUROC |
| V2 / persistent | 在 V1 上加入 copy/refresh 状态持续性，积分倾向与持续强度 | 利用位置次序，可改变局部排序 | 风险状态只用 absent 专家，容易把片段中的来源可预测词当反证 |
| V3 / witness | 保留 V2 状态链；把风险状态改成半 source、半 absent 的混合专家 | 容许同一风险区域内出现普通词，负发射证据有下界 | 没有新增语义信息；隐状态不等于事实真值；上下文可能改变乃至反转个体排序 |

V3 相对 V2 的直接算法改动集中在发射分布；持续性与精确积分来自 V2，不能再算成 V3 新增三次。

## 第一步：固定原回答，观测来源对比

对已有回答的同一 token、同一原始历史，冻结语言模型分别计算有来源与删来源的条件概率。记 source 为1、absent为0：

\[
d_t=\log q_0(y_t\mid x_{\setminus S},y_{<t})
     -\log q_1(y_t\mid x,S,y_{<t}).
\]

原回答不重新生成，不因世界变化而重新分词。有效 token 和字符 offsets 一起保留，按预先固定的原片段独立读出。分数高只意味着预测专家偏好，不自动意味着错误。

实际实验的 full-gap 采集可通过 measure_responses → counterfactual.observe_responses，两世界同用 SDPA。默认文本 API 的 Observer 还测 local/route，present eager、absent SDPA；复现某个结果必须核对该次运行路径，不能把不同路径混成一次预算相等实验。

## 第二步：V2 的持续状态先验

每个固定片段共享倾向 π 与复制概率 c；二者预先设为独立 Uniform(0,1)，不从目标标签估计。

\[
\pi,c\overset{ind}{\sim}\operatorname{Uniform}(0,1),\qquad
Z_1\mid\pi\sim\operatorname{Bernoulli}(\pi),
\]

\[
P(Z_t=1\mid Z_{t-1},\pi,c)=cZ_{t-1}+(1-c)\pi.
\]

可以理解为：概率 c 复制上个状态，概率 1-c 从共享倾向重新抽样。c=0 回到交换型共享先验；V3 此时仍使用混合发射，因此是“V3 c=0”，不能直接叫原 V1。

## 第三步：V3 的核心优化是混合发射

V2 在状态0用 q₁，状态1用 q₀。V3 改为：

\[
q(y_t\mid Z_t=0)=q_1(y_t),\qquad
q(y_t\mid Z_t=1)=\frac{q_1(y_t)+q_0(y_t)}{2}.
\]

所有概率仍以同一原回答历史为条件，此处省略。除去每个状态共有的 q₁ 因子，状态1的 log 发射变成

\[
e_t=\log\frac{q_1(y_t)+q_0(y_t)}{2q_1(y_t)}
   =\log(1+\exp d_t)-\log2.
\]

这是一个归一化专家混合推导出的似然比，不是随意把 d 截断。它具有

\[
e_t>-\log2,\qquad
\frac{de_t}{dd_t}=\sigma(d_t)>0,\qquad
e_t\approx d_t-\log2\quad(d_t\gg0).
\]

例如 d=-10 的强负证据，转换后约为 -0.693；d=3 则约为2.355。其设计动机是：一个不受支持的陈述仍会包含语法词、衔接词和来源容易预测的词，这些词不应给整段无界的负反证。**这只是预测模型中的合理假设，尚非事实识别定理。**

## 第四步：联合积分，而非平均若干平滑结果

对整段所有状态和两个共享参数积分，输出每个位置的边缘 log odds：

\[
s_t=\log
\frac{\int_0^1\!\int_0^1\sum_{z:z_t=1}
       p(z\mid\pi,c)\exp(\sum_jz_je_j)\,d\pi\,dc}
     {\int_0^1\!\int_0^1\sum_{z:z_t=0}
       p(z\mid\pi,c)\exp(\sum_jz_je_j)\,d\pi\,dc}.
\]

按原实现，myproject/sequence.py 的 persistent_scores(witness=True) 用正权 Gauss–Legendre 节点与两状态前后向求这些未归一化联合量。有限段下，对 π 的次数≤n、对 c 的次数≤n−1；默认每轴 floor(n/2)+1 个节点足以精确积分这些多项式（实际浮点有舍入误差）。不能先对每个节点归一化再均匀平均后验。

实现还给出精确分解：

\[
s_t=e_t+g_t(e_{-t}).
\]

自身发射以系数1保留；g 是 leave-one-out 上下文修正。V1 的片段保序与非负交叉影响结论不适用于完整 V3。已有有限反例允许增强某处证据反而降低另一处后验；这不等于已发现语义错误的因果机制。

默认 V3 单片段时间约 O(n³)，前向缓存也随节点网格增长。未改变语言模型，不训练，不读取评价标签。不会比只读同一批 gap 的 V1 获得更多输入信息。

## 第五步：调用及逐行代码对应

显式指定 method="witness" 使用 V3；method="persistent" 是 V2，method="shared" 是 V1，method="r4" 是旧配方。

~~~python
from myproject import HallucinationDetector

detector = HallucinationDetector.from_pretrained(
    "/path/to/Meta-Llama-3.1-8B-Instruct",
    task="QA",
    method="witness",
)
~~~

上面只是公开接口用法，不等于重现论文具体采集预算。

| 算法环节 | 唯一生产实现 | 逐行说明 |
|---|---|---|
| 固定 token、offset、分段 | myproject/inputs.py 的 encode_response、text_units | [inputs](files/myproject_inputs.md) |
| 两世界 full 观测 | myproject/counterfactual.py | [counterfactual](files/myproject_counterfactual.md) |
| 分段与方法派发 | myproject/shared.py | [shared](files/myproject_shared.md) |
| e=logaddexp(0,d)−log2 | witness_emissions | [sequence 全行](files/myproject_sequence.md) |
| 先验网格、前后向、证据积分 | _grid、persistent_scores | [sequence 全行](files/myproject_sequence.md) |
| 原字符区间输出 | myproject/measure.py 的 token_scores | [measure](files/myproject_measure.md) |

## 有效性到底支持到哪里

旧 Llama 上，FinQA 从 V1 的0.7283/0.2572升至 V3 的0.8027/0.3250，TATQA 从0.7688/0.3559升至0.8430/0.4657。这是 FRED 发布的162/130答 silver 人群上的历史探索证据，使用source-only completion提示；不代表原始FinQA/TATQA全基准，原生成问题没有恢复。
R4 在 QA 和 Data2txt 更强；Qwen 金融上 Matched local 的点值略高，配对区间跨0；Qwen HalluRAG 上 NLL / LUMINA 也能高于 V3。因此“当前保留候选”不能改写为“全部任务最优”。

后续 responsibility、onset、token_state、adaptive_token、conditional 等已有失败结果，新的代码时间戳不代表新的有效方法。当前保留V3；[无标注的识别边界](UNLABELLED_LIMITS.md)说明为何关系分支退出主线，下一步先测可操作的预测瓶颈，再决定是否形成新的评分方法。[此前组合候选](INTEGRATED_DESIGN.md)保留为未执行的受限分支，不能用既有探索标签挑选组合。
