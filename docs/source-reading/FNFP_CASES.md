# 真实FN／FP：来源事实、标注边界与失败步骤

先看原文和具体失败位置。这五个金融案例显示了不同原因：有的原始警告被上下文压掉，有的原始测量已经很弱，有的错误短语只有一部分被检出，还有合法计算被误报。不能把它们统一归因于“传播太强”，也不能据此直接提出一个加权新方法。

本页核对恢复的原始prepared输入、发布标注、完整任务分数与原token轴。精确ID、字符位置、原文短引、文件／记录／文本hash及逐token数值见[来源证据JSON](../results/fnfp_qualitative_evidence.json)。全任务数值分解和RAGTruth案例见[机制审计](FNFP_MECHANISM.md)。这些是最终评价后的事后诊断，不是新增检测结果。

## 怎样读下面的数值

原始full gap比较同一观察模型在给定原回答前缀下、有源与无源条件生成当前token的概率：

\[
d_t=\log p_{\mathrm{absent}}(x_t\mid x_{<t})-\log p_{\mathrm{source}}(x_t\mid x_{<t}).
\]

V3先把它转成自己的witness发射量，再与该模型的上下文项组合：

\[
e(d_t)=\log\frac{1+\exp(d_t)}{2},\qquad
\ell_t^{V3}=e(d_t)+c_t.
\]

这里的上下文项通过既有模型的原片段推断得到。本页用生产函数`witness_emissions`、`persistent_scores(..., witness=True)`重放原单元，与封存分数最大绝对误差不超过8.89×10⁻¹⁶；没有另写评分核或改分段。上下文项正负表示模型偏好的变化，不是邻居是否真实正确的证明；gap和潜状态分数也不是校准后的事实错误概率。

所有排名从1开始，按各自完整任务降序排序，同分按原始顺序稳定打破。TATQA共有7747个token，top5%预算388；FinQA共有16047个token，预算803。分数大于0仍可能跌出预算而成为FN。

| 案例／token | 原始gap | 自身发射量 | 上下文项 | V3最终分数 | 原始排名→V3排名 |
|---|---:|---:|---:|---:|---:|
| TATQA54：` Acc` | 2.650698 | 2.025772 | −2.167958 | −0.142186 | 173→2866 |
| TATQA51：` monthly` | 3.914544 | 3.241150 | −2.361789 | 0.879361 | 92→1068 |
| FinQA53：`8` | 0.368423 | 0.201084 | −0.106226 | 0.094858 | 1690→5810 |
| FinQA59：` revenue` | 4.133605 | 3.456356 | −1.836436 | 1.619920 | 127→785 |
| FinQA59：` recognition` | −4.947249 | −0.686069 | 0.472553 | −0.213516 | 15243→7810 |
| TATQA3：` million` | 5.424751 | 4.736001 | 1.000846 | 5.736846 | 45→28 |
| TATQA3：` impairment` | −0.000532 | −0.000266 | 1.898902 | 1.898636 | 2134→302 |

## TATQA54：表格科目错了，首子词的警告确实被压掉

**ID：`fred:tatqa:test:54`。** 来源表格的两个明确标题是：

> Deferred tax assets
>
> Deferred tax liabilities

回答却写成：

> The two main items measured in this table are Accrued expenses and Deferred tax liabilities.

第二个科目仍与表格一致，第一个把“递延所得税资产”替换成“应计费用”。这不是需要复杂算术才能确认的错误，直接核对表格标题就能定位。

发布的entity编辑跨度为`[46,62)`，原文是`Accrued expenses`，覆盖四个原始子词。首片段` Acc`的token字符范围为`[45,49)`，因为包含前导空格而与标注边界略有不同，仍按字符重叠标正。

首片段的原始排名173本来在388个告警名额内；自身发射量2.025772被−2.167958的上下文项抵消，最终排名2866，成为FN。**这个具体token有可复核的上下文压制。** 但另三个子词的原始证据本来就弱或方向相反，不能声称四个FN全由传播造成；它们也不是四个独立语义错误。

## TATQA51：错的是覆盖期，不是开票频率

**ID：`fred:tatqa:test:51`。** 来源原句：

> For multi-year agreements, we generally invoice customers annually at the beginning of each annual coverage period.

回答：

> Multi-year agreement customers are generally invoiced annually at the beginning of each monthly coverage period.

两者都说每年开票，回答把**每个annual coverage period**改成了**monthly coverage period**。所以准确的错误描述是覆盖期时长关系annual→monthly，不能写成“每年开票变成每月开票”。

发布relation跨度`[88,95)`为`monthly`，对应原token` monthly`的`[87,95)`。原始排名92在预算388内；上下文项−2.361789把V3排名压到1068。最终分数仍为正，也已经漏报。这是第二个明确的原始警告被上下文压掉的例子。

## FinQA53：金额和方向都不对，但标注只覆盖编辑金额

**ID：`fred:finqa:test:53`。** 来源表格：

| 项目 | 金额，百万 |
|---|---:|
| 2014 net revenue | 5735 |
| 2015 net revenue | 5829 |

来源支持的变化是5829−5735＝**增加94百万**。回答却说：

> ...was a decrease of $8 million ($5829 million - $5735 million).

它抄对了减法的两个操作数，却把结果写成8，并把方向写成decrease。发布的numerical编辑跨度只标了`[84,94)`的`$8 million`，没有把`decrease`标正。**未标正并不能证明该词语义正确；这里的指标评价的是发布的编辑跨度。** 如果检测器指出方向有误，不能仅因方向词位于编辑跨度之外就断言其语义判断错误。

`8`的原始排名1690本来就不在预算803内，说明原始测量没有形成足够强的告警。V3又把它降到5810，确实进一步削弱了证据，但不能把最初的漏报全部归因于上下文传播。这个例子首先暴露了概率对比没有完成数值／方向事实核验。

## FinQA59：算式对了，账户错了；一个短语出现TP和FN

**ID：`fred:finqa:test:59`。** 来源明确介绍的是：

> ...gross liability , excluding interest and penalties , for unrecognized tax benefits

原表数值给出：53818−36528＝17290，且期末195237−期初177947也等于17290。回答却把对象写成：

> In 2014, the net change in revenue recognition was an increase of $17,290.

数字差值吻合，不代表“收入确认”这个科目正确；原表谈的是未确认税收利益对应的负债变化。当前显示的来源片段没有清楚的金额单位标题，本页只确认表内数值差，不额外断言美元尺度正确。

发布entity跨度`[27,46)`是`revenue recognition`。` revenue`原始排名127，经过负上下文项后排名785，仍在803预算内，因此是TP。` recognition`的原始gap却为−4.947249，排名15243；上下文项反而把它提高到7810，但仍是FN。

**后半个词的漏报不是上下文压制。** 它在原始测量阶段已经方向错误，上下文实际上有所改善。逐token条件概率还依赖已有回答前缀；错误短语的后续子词不能被当成独立的整句语义核验结果。

## TATQA3：合法计算被误报，上下文又扩散到正常词

**ID：`fred:tatqa:test:3`。** 来源表头为`USDm`；“Net profit/(loss) for the year excluding impairment”一行是：

| 年份 | 2019 | 2018 |
|---|---:|---:|
| 不含减值的净利润／亏损，USDm | 46.0 | −34.8 |

回答计算：

> 46.0 - (-34.8) = 80.8
>
> ...was an increase of 80.8 USD million.

这与表格的数值、变化方向和单位一致。80.8没有作为原始单元格出现，但它是合法推导；`million`是表头`USDm`对应的单位表达。发布标注跨度为空，本页也通过来源和算术独立确认该例，而不是只依靠“未标错”。

原始模型对` million`给出高gap5.424751、排名45，已经会误报；正上下文项又把它抬到排名28。更明显的扩散发生在正常的` impairment`：原始gap约−0.000532、排名2134，经1.898902的正上下文项变成排名302，进入告警预算。该答最终出现14个V3误报token。

这里同时存在**原始测量对合法表达误报**和**上下文把正常邻居抬高**。不能把“不是逐字复制来源”“token概率变低”或“潜状态风险高”直接等同于事实不受支持，也不能仅凭这一个例子推断模型内部为什么偏好某个单位表述。

## 这些案例支持什么，下一步怎样验证

证据能定位不同失败步骤：TATQA54／51有明确上下文压制；FinQA53原始告警已经不足；FinQA59后半短语在原始测量中方向错误；TATQA3有合法推导误报及误报扩散。只改传播强度无法同时证明这些问题被解决。

接下来先把[全任务机制统计与RAGTruth案例](FNFP_MECHANISM.md)对应到具体来源，区分原始测量、子词／条件前缀、上下文和编辑标注范围。现有分数只是来源相关的预测量，缺少的语义能力需要独立验证，不能靠公式换名或直接加权宣称补齐。

若之后研究冻结模型核验器，必须先固定输入、来源引用、输出粒度、unknown处理及全部单元覆盖，再分别评价核验器单独和完整系统、做删除对照。该模块当前仍为**PROPOSED_NOT_RUN**，本页没有新启发式分数、新方法效果或普适模块结论。

[证据与hash](../results/fnfp_qualitative_evidence.json) · [全任务机制审计](FNFP_MECHANISM.md) · [历史错例与统计](PERFORMANCE_REVIEW.md)
