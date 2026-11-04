# 独立理论/先例条件意见

本意见是对root提出的新head–carrier关联干预的理论与prior审查；formal文稿由本agent整理，不冒充第二位代码作者的独立review。另需source reviewer对最终合同fresh科学审查。0模型、0GT、0新分数。

**结论：有限计算机制 conditional GO，standalone detector NO-GO。** 读出只作原token索引的物理消息诊断。必须先过source-only闭合许可和同义重复负控、原q/all48×32与数值门，失败则0机制计算/保留unknown；不得因此重开旧source-ablation或H018strong-sign。

理论理由：同一q、同一GQA组内两head共享K/V与mask，故2×2 rank-one独立化有明确双边marginal保持的干预含义；不是从相似度猜一个关系。共享V下的局部消息差是κ(O2−O1)(va−vb)，能把head-address关联、上下文化payload和输出投影区别开。sameV/sharedO的精确零不是语义证据。标准最大熵/互信息和2×2代数不支持truth符号。

attention双比值在共同可见正权、同K、同head维度和零mask项时消去行softmax；Llama必须用RoPE后的Q/K。FP16/BF16舍入和zero使其只能作为有限/被censor的诊断；不加epsilon。换不同Rprefix或query时刻可产生无语义的query/key交互；同q GQA解决了自然单relation没有两query槽的问题，但仍有key位置和head-specific RoPE偏好。它测head×carrier association，不是数学定义上的source ownership truth。

先例实际检索与结论：

- [Counterfactual Pairing Cycles,2026 §2–3](https://arxiv.org/html/2610.23248v1) 已给加性边际消去、最小2×2和bilinear差向量分解，并明确不能排除非加性交互伪象。本方案不能把χ恒等式当新理论。
- [Mixing Mechanisms,原文§3](https://arxiv.org/html/2510.06182) 用使位置、词汇、指针机制给不同答案的干预区分机制。它支持需要区分性预测，不支持所有attention差都叫binding。
- [Unpack v3 §3/5](https://arxiv.org/html/2605.23393v3) 已有native graph、K/Q/V composition和output attribution，明确贡献不等于interventional effect。本方案的新观察只能是所指定的双边association擦除；不能宣称首次完整图或first token-conditioned attribution。
- [CoCo](https://aclanthology.org/2021.findings-emnlp.10/)已做source-counterfactual一致性。typed-source-permutation的LM likelihood mixture与本地P021 latent-world家族重叠，改名Bayes factor不能带来新观测。只有显式假设“错绑分布=先按已知语义pi取world再按该LM生成”、已知pi先验、模型分布正确，才可叫该模型的BF；这不是事实truth BF。
- [DICD出版社摘要](https://www.sciencedirect.com/science/article/pii/S0950705126015455)已有trigger sensitivity/entity specificity/tuple decomposition。全文仍403，不能声称与typed-BF逐公式等价，但干预特异性原则不是空白。

本地限定旧源码实际审查：ownership.source_context_delta已有固定A换V(X)、逐head source-mass守恒E及XE；旧endpoint swap/V donor实际存在，正是前一版δ± payload方案被否决的原因。source reviewer当前未见明确跨head同时保列量的B=rcᵀ/M/断言；generic API能表达不代表已实做。此结论仅限已审旧配置，不是全球新颖性证明。

最小可前进路径是MINIMAL_CONTRACT的M0→M1：先冻结cook/simmer/grill/onion的完整source action-duration licence及原数字q；再做全48×32独立关联擦除，不读旧effects/GT。若SEM与EQUIV缺乏差别或norm解释同等，停止语义推进；若无位置控制则只计计算pilot。后续最多两个合法source版本、同原R的保义换序/绑定变更，须另审另claim，才可尝试排固定slot捷径。独立I02语义前端是后续结构接口，不在本次实现；缺乏文本核验之外的定位/可靠性/成本增量就不晋升检测器。
