# H015：唯一 causal context 的 completion 概率树（纯数理资格）

2026-11-03。当前仅固定接口、纯标准库 kernel 与 synthetic 数学/结构测试。**不实现或启动模型 backend，不重算 H013 原32条，不改其门槛，不改旧结果。** 已报告的一题 pool mass 1.000516 与共享 prefix 的 LSE 不同作为旧仪器工程失败保留；仅靠相同 padding 不保证共享 causal context 采用同一分布，也不能从该差异单独证明硬件因果。

本合同定义新的 observer 概率树，不宣称与 LM 理想实数计算/旧 batched forward exact replay，不声称 factual correctness、自然覆盖或排名改善。H014 仍仅处理已封存 H011，H013 的新32raw未由本任务重放或评价。

## 接口与输入边界

唯一新模块 `experiments/research/completion_trie.py`，纯标准库，无 Torch/HF/tokenizer/文件 I/O：

```
build_trie(candidate_metadata: Mapping[str, completion_metadata]) -> trie | unknown
score_trie(trie, node_observations: list[observation]) -> scores | unknown
```

candidate 是来源定义的原始字符串；规范顺序只是 raw string 排序，不读 R 值或 gold。单个 trie 对应一个固定 prefix/模型/完整候选域；后续 outer 必须另外冻结模型、revision、tokenizer、运行时与数值政策，不能跨仪器复用节点。此纯模块不自行选择候选或生成提问。

输入 metadata 完全遵循现有 `completion_metadata` / collector `_META` 的18字段 v1合同：prefix P非空，candidate K非空，无外侧空白、不改原文；input=P+K，不在input追加EOS；targets=K+[EOS]，positions=P−1…P+K−1。flags、计数、schema、whitespace policy、EOS和ID型别严格验证。所有候选必须同 prefix_ids / EOS。不同原串若产生同一token路径，作为 alias unknown，不能把同一事件相加两次。

为保持纯标准库，新文件声明该已冻结 v1 metadata字段/常量，不导入带HF类的旧collector/answer_completion模块；测试将直接比对现有 `_META` 及实际纯 framing helper输出，不能静默创造不兼容 framing。

## 唯一节点与 EOS prefix-free 路径

对候选 K=(k0,…,kK−1)，其第i个target（i=0…K）使用完整 causal context：

```
context_i = prefix_ids + candidate_token_ids[:i]
target_i  = candidate_token_ids[i] if i < K else EOS
```

context整数列表完全相同才是同一node；节点key由contract与完整context hash产生。每node只保留一组按token ID排序的children，每条edge为一个唯一目标token。每candidate保存从根到固定EOS edge的完整path。EOS不能在任何candidate K内，同一EOS用于所有候选，EOS edge没有后继。

当一个候选名是另一候选名的token prefix时，共享节点同时有 EOS edge（短名终止）与另一个非EOS edge（长名继续）；二者必须读取**同一行完整vocab logits和同一个normalizer**。有EOS终止的候选code彼此prefix-free。相同context不允许为不同candidate各保留一份row。

trie保留输入metadata、canonical nodes/edges/paths、候选序列和logical target positions / unique node / unique edge计数。scorer先用保留metadata重建并exact核对trie结构，不接受手工重连、增删节点或候选path。

## observation 与数值政策

纯 scorer 输入每node恰一个record：

```
{node_id, context_ids, vocab_size, full_vocab_logits: [finite Python int/float, ...]}
```

record population、node ID、context、vector rank/长度、vocab大小和target ID界限先完整校验；缺node、多node、重复node/context、重复edge定义、EOS或framing异常均显式unknown。所有node同一vocab_size，bool/NaN/Inf/不可转换数字拒绝。完整vector仅为本次数学函数输入，**不放进返回结果**。

每row用Python float64和stable shifted logsumexp：m=max(logits)，s=fsum(exp(logit−m))，z_shift=log(s)，fullV_logZ=m+z_shift。每edge logp=(edge_logit−m)−z_shift，避免极负共同偏移下绝对logZ舍入抵消。返回m、z_shift、fullV_logZ、所需edge原logit/logp/probability，使raw sum可按明确政策复核；不把stable计算差异隐藏为校准。

固定采用**无epsilon、无clamp**的严格门：任何positive edge logp、selected-child log mass >0 / mass >1、nonfinite派生量、candidate pool log mass >0 / mass >1均unknown。有限精度下本来合法的row也可能因此被工程拒绝；保留拒绝，不调阈值或修rank。probability underflow=0只标明浮点表示；有限log joint、log pool mass和稳定条件质量仍单独保留。若算术overflow不能形成有限raw logsum则unknown，不能用假0风险代替。

每candidate的 `raw_joint_log_probability` 是沿唯一node rows的完整K+EOS edge logp之和（math.fsum）；pool log mass使用全部候选raw joint的stable LSE，pool mass及underflow另存。conditional mass只表示给定这个完整候选集合的条件质量，明确不是full-vocab probability或事实后验；不通过候选softmax倒推/替换raw joint。

## 数学资格与能证明的范围

在精确实数、同一node归一化分布p下，每个children集合的概率和≤1。对树自底向上，令一个node下属于候选code的终止质量为T：EOS edge贡献其p，非EOS edge贡献p×子树T；由子树T≤1及children和≤1可得T≤1。根T即所有完整候选EOS事件之和，因此pool mass≤1。

这是**一致概率树的归纳性质**；若不同候选在相同context使用不同row，它们不能拼成同一概率树，定理不适用。finiteprecision具体结果仍须上述严格门，不把实数定理冒称机器舍入必定满足exact界。

纯scorer看到了full vector，能独立计算该输入row的normalizer。未来真实backend若只保存edge logits+fullV_logZ/分解，则CPU reader只能核保存值的一致性与backend provenance，无法验证未保存的全部150k logits；必须明示这一attestation边界。不能用旧candidate forward里看起来接近的prefix rows合成一次真实shared-node观察。

## 未来工程不在本轮执行

root提出的原则性backend是：每个唯一prefix-only context单独forward一次，input_ids不含future candidate token，`logits_to_keep=1`，从这一row同时读取所有children token的logit与full-vocab float64 LSE。后续也可预先冻结等价批处理，但不把right padding本身当成一致性证明。工程方案需root/lead审查后另行资格，不自动采用。

本轮只报告structural node counts和synthetic资格，不把unique nodes称实际forward、不给运行速度/显存节省或新模型效能；旧H013的32次成本不改写。

## Synthetic 测试与停走

用现有framing helper+假tokenizer构造multi-token候选、同前缀不同名、短名是长名prefix的EOS分支、多个共享token。full-vocab fake向量提供可独立枚举的oracle；核每个节点只出现一次、每path完整K+EOS、pool mass≤1、raw sum/conditional mass与oracle一致。

拒绝missing/duplicate/extra node，重复child/被改trie，candidate token alias、prefix/EOS不一致或candidate内部EOS、bool/negative/out-of-vocab IDs、rank2/wrong-length/非finite vector、normalizer溢出与结构别名。另测极负共同logit偏移、log-space underflow保留，及明确概率门失败时不clamp。

停止点：纯kernel/tests与独立review完成，只表示observer定义可审、概率树数学合同通过；没有backend admission、真实32重算、GPU launch、QA准确率或主方法promotion。
