# H017 M3：角色×数值交叉的地址/内容路径检验

2026-11-03，编码前科学准入。Root已接受四来源及48次模型调用预算，待inventory peer独立审查。只研究冻结Llama观察器如何处理同一份预先提供的回答R；不读目标标注、不设计风险融合、不恢复角色QA，也不宣称新颖性或检测提升。

## 唯一主要问题和来源许可

在R数值尚未读入的预测位置，来源归属变化是否通过Q/K改变检索地址及最终输出，而数值变化能否通过V传递、保持该层attention不变？这是区别于整体source mass、cosine或删除指纹的新干预问题。

来源严格采用构造schema：`{"revenue_by_company": {"Alpha": 100, "Beta": 200}}`。字典键是公司主体，数值是该公司收入；只允许两个键、100/200、无时间/否定/推导。语义由构造规则许可，不是自然QA解析能力或M0 prose unknown的升级。

| 来源 | 字段顺序 | 对固定R中Alpha—200的支持 |
|---|---|---|
| S00 | Alpha100；Beta200 | 不支持 |
| S10 | Beta100；Alpha200 | 支持；只换主体，数值及位置不变 |
| S01 | Alpha200；Beta100 | 支持；只换数值，主体位置不变 |
| S11 | Beta200；Alpha100 | 不支持；S00完整事实保义重排 |

S10/S01也是保义重排对。四来源的主体/数值词袋相同，关系取决于role×value交互。固定任务为`Summarize the revenue records faithfully.`后附唯一来源；固定R=`Alpha revenue was 200.`。模型只看到R中200之前的原token前缀。R在任何模型观察前固定，不让模型另答角色问题，不汇报生成问答正确率。

## 必须在模型调用前通过的token资格

- 用同一序列化器保持字段顺序，复用主线encode_source/encode_response、typed_source/source_inventory，保留完整路径及两个数值occurrence的原字符区间。
- 完整R的编码定义200的唯一数值token和接收位置；100候选的完整R编码必须具有完全相同的预测前缀。100/200各恰好一个数值token。不能对截断R重新分词改轴。
- 四世界的完整模型前缀长度、R token前缀、prevalue位置、两个来源数值token位置完全相同；role donor在对应位置的数字ID也完全相同。数值必须是完整单token，不强行分摊跨界BPE。输入≤256、batch1。
- 条件端点为R中Alpha的末token，必须早于prevalue且位于prefill；同时固定两个source之前、非special/非主体/非数字的instruction token作为off-target位置（从左到右前两个）。
- 若Alpha/Beta分词、数字边界或位置不合格：eligibility失败、0模型调用，不换名字/格式/措辞自动救援。所有资格、IDs、offset、规则在原模型前向前冻结。
- 原始9资产与prefix、full-head/cache/no-op门必须已合格。开发只运行官方tiny CPU合成测试；root独立claim真实运行。

## 唯一模块与一个必要内容对照

全部32层执行到最终原生head；干预只在固定block15。Q用全部32物理头，K/V用全部8KV头，两个来源数字位置全部使用；不选最佳head/层/occurrence。

Recipient S(r,v)的role donor为S(1-r,v)，value donor为S(r,1-v)。每个recipient固定10个branch：

1. Q-role：prevalue接收位置的raw pre-RoPE Q换成role donor。
2. K-role：同一绝对位置的两个数字post-RoPE K换成role donor，V保留recipient。
3. V-role：同两位置V换成role donor，Q/K保留recipient。
4. QK-role：上述Q+K。Llama不要求该交互相消，相消不是成功门。
5. V-value：同两位置V换成value donor，Q/K保留recipient；这是必要的内容通道对照。
6. identity no-op：Q.clone以及两个数字位置自身K/V复制，必须与baseline最终hidden/full-vocab logits/logp逐位相同。
7. off-target random-K：seed0随机K变化放在固定两个非role/nonvalue位置；每KV头/每位置L2范数和数量匹配K-role变化。它是数值damage control，不认证语义无关、位置等价或跨位置role因果。
8–10. on-target random-K：seed0/1/2独立各向同性方向、同样每KV头/位置范数，在两个数字位置分别执行。零范数行保持零，不挑seed。

所有branch从独立pristine全32层prefix KV clone开始；donor/pristine不能alias/被改，query append不能继承前branch。高层source KV保持recipient，因此混合世界不是重建完整donor。K为cache的post-RoPE，Q hook为pre-RoPE；Q由原生模块在真实位置旋转。严禁rawQ点乘rotatedK，也不跨位置复制donor K假称等价。

## 理论和固定观测终点

局部精确关系为`o=a(Q,K)V`，K-only的变化为`[a(Q,K')-a(Q,K)]V`；V-only的变化为`a(Q,K)(V'-V)`。K-role保留原V，其路径起点是地址权重；V-value保持该层Q/K/attention属于仪器恒等式，不自动证明V只编码数值。Donor K/V仍混合上下文化信息；最终输出因果必须经完整网络测量，不能用局部AV代替。

每个baseline/branch保存完整128256词表logits、最终hidden、原200的原始归一化logp/概率、100/200的概率与rank、两者总概率质量及margin `M=z200-z100`。两候选都极不可能时，margin偏移只有局部读出意义。预定task-legibility字段为四baseline top1是否都在{100,200}；不满足不升级语义解释，也不选成功世界重跑。高置信错误表达仅在不支持R的来源中原p200>0.5时描述，其余照报。

同批前向保存condition-end和prevalue全部Q32，condition行从prefill钩子取，不增加调用。postvalue不在预算，不能声称已验证回声/持续性。完整保存全部32头在两个数值occurrence的attention及总source mass，不能只输出平均或最大head。cosine、集中度、热图不是真实性或新风险读出。

**新的post-RoPE instrument合同独立资格，不借旧raw-cache门冒充：** 同一前向捕获实际postpatch rawQ、原生position_embeddings(cos/sin)和block15 o_proj输入(native headAV)。Q patch hook先于Q capture hook。使用官方apply_rotary_pos_emb和原生eager attention算子，对当前Q按实际绝对位置旋转，再读取当前branch cache的post-RoPE K/V，保持连续GQA与每个接收位置的causal mask；不改变实际模型SDPA前向。condition行必须屏蔽prefill中其后的key；prevalue含当前token self key。将重放AV全部坐标与native o_proj输入比较，固定容差`0.002+0.0001*abs(native)`；不匹配为instrument failure并停止。此过程0额外model/head调用。Tiny官方模型测试须拒绝遗漏RoPE、错误GQA和未来mask。完整Q/K/attention在V-role/V-value臂必须与该recipient baseline逐位相同。

## 不同解释与固定科学门

- 归属/地址：K-role应把层15检索偏向recipient内counterfactual目标所在数字位置，最终margin朝该数字移动，而原V没有新内容。四recipient同时提供正反方向及两种value赋值。
- 内容：V-value保持层15Q/K/attention，最终margin可朝该地址下的新payload移动。只有内容对照有效而K-role无选择性效果，则拒绝本block15数字K归属解释。
- 查询/读出：Q-role强而K-role弱支持接收条件/检索状态的替代解释；V-role强但attention不变支持上下文化V/后续读出。结果均保留，不事后换主假设或挑最佳组件组成检测器。
- 非特异损伤/位置：random同等大，或保义重排差异超过角色效应，均削弱归属解释。

对每recipient，令target是其原来源中与role donor的Alpha数值相同的唯一数值occurrence，other是另一个。保存每头`a(target)-a(other)`；固定地址读出`A=mean_all32[a(target)-a(other)]`，无条件全32头，不另行归一化数字质量。Primary为四recipient方向一致的K-role margin变化；方向由role donor中Alpha的许可数值固定。工程deadband固定为`0.008+0.0001*(|z100_base|+|z200_base|+|z100_patch|+|z200_patch|)`，只作数值决策容差，不称网络误差定理。四个角色效应均须超过该deadband、各自3个on-target随机K最大绝对margin变化及off-target变化，同时`A_Krole-A_baseline > 1e-5`。这个attention deadband预定为工程判别范围，不是理论泛化界；source/数字总mass另外报告，不能代替A或按头选择。

保义布局门：`max(|M00-M11|, |M01-M10|) < min(四个方向化K-role效应)`。V-value对照须保持该层Q/K/attention且四recipient margin按其value donor方向超过同样deadband。单quartet只支持可证伪的局部可行性，不把4recipient或3seed当独立自然人口统计。任一门失败保留竞争解释/unknown，不调规则、head、层或prompt。

## 固定预算、复用和失败处理

四source各1次prefill+1次baseline last-query：8次model.model/4次head。四recipient各10branch：40次model.model/40次head。**总48次model.model、44次完整head，其中44次last-query。** 不额外生成回答、采postvalue、多跑seed或层。身份完全相同的完成单元可封存复用；只改分析不重采。科学negative完成剩余预定分支；身份/no-op/资源失败停止、保留实际计数，不重试/扩预算。

代码仅experiments/relation_intervention.py及必要的experiments/research/relation_intervention.py，复用主线输入、已合格streamer、head/cache、attention kernel、身份/seal；root拥有CLI/打包/持久状态。先封存全部source/R/位置/branch/seed/参数/后端/代码，再root claim。代码开发不是原始模型运行许可。

## 与旧路线及结论边界

这里是同一提供R在Q/K/V路径上的原生最终token干预；不是P002距离、P003删除、P005post-token方向、P021行列null混合或P026repair/reset，不训练目标探针、不恢复H015/H016角色答题。地址/内容分离有已有绑定论文先例，不能称其本身新颖。

即使通过，也只支持构造许可域、当前观察器/层/位置的role-sensitive mediation；位置ID/binding-ID策略也可能通过单quartet，不能推断唯一语义算法或电路。若baseline已拒绝所有错误表达，只能重现正常检索，不能解释自然FN；简单JSON规则在此域已能直接判定支持，不能据此宣称白盒必要或低成本。M0自然QA仍unknown。自然检测仍须source-only许可、原回答全轴风险/unknown接口、冻结全部分数后的评价，以及同预算文本核验；当前没有新AUROC结论。
