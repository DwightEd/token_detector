# myproject/distribution.py 逐行讲解

基线：`1210120748b3a05bd271bc79debcbed4fdc0c964`；SHA-256：`891b83c1f5c48dbf9926ac211aaca1c5d5a57942a6b392b86de5ac95104d7bb2`；106 行。

源码与注释逐行并排。一个跨行表达式共用同一操作说明；这不代表各行独立执行。完整调用路径、输入输出轴和理论边界见 [METHOD.md](../METHOD.md)。未执行的 GPU 路径见 [验证边界](../VALIDATION.md)。

| 函数（含嵌套） | 原始行号 | 作用与契约入口 |
|---|---|---|
| `log_calibrator` | 11–18 | log_calibrator 输入 log u≤0，返回 log∫_0^1 a*u^(a−1)da；由尾质量转换调用，不是用标签训练的校准器。 |
| `conditional_observations` | 21–73 | conditional_observations 输入匹配 [Q,V] 两世界 logits 和目标 IDs[Q]，输出八条 [Q] 诊断；所有词表原子参与，不只 top-k。 |
| `conditional_observations.<lambda>@63:13` | 63–63 | 嵌套 lambda gather 捕获目标下标，从任何 [Q,V] 数组提取实际 token 的 [Q] 值。 |
| `SourceDistributions.__init__` | 79–83 | 构造器输入原回答 IDs，保留长度和每块目标切片。 |
| `SourceDistributions.present` | 85–89 | present(start,logits) 是 source_full 的回调，输入原生 logits 块。 |
| `SourceDistributions.absent` | 91–101 | absent(start,logits) 配对同起点的 present 块，归约后释放暂存。 |
| `SourceDistributions.finish` | 103–106 | finish 在两个世界观测完成后返回全回答诊断；由 Observer 调用。 |

| 行 | 原始代码 | 操作、目的与边界 |
|---|---|---|
| 1 | <code>"""Conditional predictive calibration; no factuality calibration is assumed."""</code> | 条件预测校准模块；不假设做过真实事实标签校准。 |
| 2 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 3 | <code>import numpy as np</code> | NumPy 用于将各词表块归约后的 CPU 数组拼成回答。 |
| 4 | <code>import torch</code> | PyTorch 在 logits 所在设备上执行 FP64 词表归一化和排序。 |
| 5 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 6 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 7 | <code>DISTRIBUTION_FIELDS = ('calibrated_gap', 'rank_gap', 'source_tv', 'source_tail_logp', 'distribution_gap',</code> | 固定八个诊断字段名，供 Observer/answer_rows 共享可选 schema。 |
| 8 | <code>                       'bridge_gap', 'source_nll', 'source_entropy')</code> | 固定八个诊断字段名，供 Observer/answer_rows 共享可选 schema。 |
| 9 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 10 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 11 | <code>def log_calibrator(log_u):</code> | log_calibrator 输入 log u≤0，返回 log∫_0^1 a*u^(a−1)da；由尾质量转换调用，不是用标签训练的校准器。 |
| 12 | <code>    """Log integral_0^1 a*u**(a-1) da, stable at u=1 and tiny u."""</code> | log_calibrator 输入 log u≤0，返回 log∫_0^1 a*u^(a−1)da；由尾质量转换调用，不是用标签训练的校准器。 |
| 13 | <code>    length = -log_u</code> | 令 length=−log u≥0，便于处理 u 接近1或极小的情况。 |
| 14 | <code>    small = torch.log(.5 + length/6 + length.square()/24</code> | 小 length 用积分的泰勒展开 .5+length/6+…，避开两个近数相减导致的消失误差。 |
| 15 | <code>                      + length.pow(3)/120 + length.pow(4)/720)</code> | 小 length 用积分的泰勒展开 .5+length/6+…，避开两个近数相减导致的消失误差。 |
| 16 | <code>    safe = length.clamp_min(1e-3)</code> | 一般分支把 length 下限设1e−3，使即使 torch.where 同时计算两分支也不在0处除0。 |
| 17 | <code>    large = safe + torch.log1p(-(1+safe)*torch.exp(-safe)) - 2*torch.log(safe)</code> | 使用闭式 log 积分 length+log(1−(1+length)e^(−length))−2log length。 |
| 18 | <code>    return torch.where(length &lt; 1e-3, small, large)</code> | 按 length 阈值逐元素选稳定分支；调用方负责合法 log u。 |
| 19 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 20 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 21 | <code>def conditional_observations(source_logits, absent_logits, targets):</code> | conditional_observations 输入匹配 [Q,V] 两世界 logits 和目标 IDs[Q]，输出八条 [Q] 诊断；所有词表原子参与，不只 top-k。 |
| 22 | <code>    """Reduce full-vocabulary logits to target scores, retaining complete ties.</code> | conditional_observations 输入匹配 [Q,V] 两世界 logits 和目标 IDs[Q]，输出八条 [Q] 诊断；所有词表原子参与，不只 top-k。 |
| 23 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 24 | <code>    Q_rank(v) = p_source(v) h(u(v)) / E_source[h(u)], where u is the source-law</code> | 先按来源-law 对 gap 排序得上尾质量 u，再构造归一化 Q_rank，最后按两世界 TV 与来源律混合，来源效应消失时证据消失。 |
| 25 | <code>    upper-tail mass of log(p_absent/p_source). The final alternative mixes</code> | 先按来源-law 对 gap 排序得上尾质量 u，再构造归一化 Q_rank，最后按两世界 TV 与来源律混合，来源效应消失时证据消失。 |
| 26 | <code>    Q_rank with p_source using TV(p_source,p_absent), so vanishing source</code> | 先按来源-law 对 gap 排序得上尾质量 u，再构造归一化 Q_rank，最后按两世界 TV 与来源律混合，来源效应消失时证据消失。 |
| 27 | <code>    effects yield vanishing evidence at each fixed positive-probability atom.</code> | 先按来源-law 对 gap 排序得上尾质量 u，再构造归一化 Q_rank，最后按两世界 TV 与来源律混合，来源效应消失时证据消失。 |
| 28 | <code>    h is the Vovk--Wang (2021), Eq. (2), p-to-e mixture calibrator.</code> | p-to-e 变换与 FP64 归一化是声明的预测分布构造；不沿用其他读出历史 FP32 的目标似然算术，不等同事实概率。 |
| 29 | <code>    All vocabulary atoms participate.</code> | p-to-e 变换与 FP64 归一化是声明的预测分布构造；不沿用其他读出历史 FP32 的目标似然算术，不等同事实概率。 |
| 30 | <code>    Double precision normalizes both worlds consistently, independently of the</code> | p-to-e 变换与 FP64 归一化是声明的预测分布构造；不沿用其他读出历史 FP32 的目标似然算术，不等同事实概率。 |
| 31 | <code>    historical FP32 target-likelihood arithmetic used by the other readouts.</code> | p-to-e 变换与 FP64 归一化是声明的预测分布构造；不沿用其他读出历史 FP32 的目标似然算术，不等同事实概率。 |
| 32 | <code>    """</code> | p-to-e 变换与 FP64 归一化是声明的预测分布构造；不沿用其他读出历史 FP32 的目标似然算术，不等同事实概率。 |
| 33 | <code>    if (source_logits.ndim != 2 or source_logits.shape != absent_logits.shape</code> | 要求两矩阵同形、二维且词表非空，否则无法按同一个原子集合比较。 |
| 34 | <code>            or source_logits.shape[1] == 0):</code> | 要求两矩阵同形、二维且词表非空，否则无法按同一个原子集合比较。 |
| 35 | <code>        raise ValueError('Expected matching nonempty vocabulary matrices')</code> | 要求两矩阵同形、二维且词表非空，否则无法按同一个原子集合比较。 |
| 36 | <code>    targets = torch.as_tensor(targets, device=source_logits.device)</code> | 把目标移到 source 设备；保持输入 dtype，后面必须为 torch.long。 |
| 37 | <code>    if (targets.shape != (len(source_logits),) or targets.dtype != torch.long</code> | 每行一个合法词表下标，形状[Q]、int64、0≤id<V；拒绝浮点 ID。 |
| 38 | <code>            or torch.any(targets &lt; 0) or torch.any(targets &gt;= source_logits.shape[1])):</code> | 每行一个合法词表下标，形状[Q]、int64、0≤id<V；拒绝浮点 ID。 |
| 39 | <code>        raise ValueError('Invalid target token axis')</code> | 每行一个合法词表下标，形状[Q]、int64、0≤id<V；拒绝浮点 ID。 |
| 40 | <code>    if not torch.isfinite(source_logits).all() or not torch.isfinite(absent_logits).all():</code> | 全部 logits 有限，拒绝 NaN/Inf（包括某些被 mask 为 −Inf 的非此协议输入）。 |
| 41 | <code>        raise ValueError('Logits must be finite')</code> | 全部 logits 有限，拒绝 NaN/Inf（包括某些被 mask 为 −Inf 的非此协议输入）。 |
| 42 | <code>    log_s = source_logits.double().log_softmax(-1)</code> | 两个世界都转 FP64 后在 V 轴 log_softmax，统一预测律精度。 |
| 43 | <code>    log_0 = absent_logits.double().log_softmax(-1)</code> | 两个世界都转 FP64 后在 V 轴 log_softmax，统一预测律精度。 |
| 44 | <code>    gap = log_0-log_s</code> | gap[Q,V] 是每个词表 token 的 absent/source log 比。 |
| 45 | <code>    # The log-normalizer difference is constant across vocabulary atoms.</code> | 归一化常数对同一行词表原子是共同项；直接按原生 logit 差排序可保留精确 ties，避免两个归一化值分别舍入后拆 ties。 |
| 46 | <code>    # Order native differences directly: subtracting normalized log-probabilities</code> | 归一化常数对同一行词表原子是共同项；直接按原生 logit 差排序可保留精确 ties，避免两个归一化值分别舍入后拆 ties。 |
| 47 | <code>    # separately can split exact BF16/FP32 ties by a few FP64 rounding units.</code> | 归一化常数对同一行词表原子是共同项；直接按原生 logit 差排序可保留精确 ties，避免两个归一化值分别舍入后拆 ties。 |
| 48 | <code>    ordering = absent_logits.double()-source_logits.double()</code> | FP64 中减原生 logits 获相对次序键，仍为 [Q,V]。 |
| 49 | <code>    ordered, order = ordering.sort(dim=-1, descending=True, stable=True)</code> | 逐行降序稳定排序，order 保存词表 ID。 |
| 50 | <code>    sorted_s = log_s.gather(1, order)</code> | 按同一 order 重排来源 log 概率。 |
| 51 | <code>    log_tail = sorted_s.logcumsumexp(-1)</code> | 沿降序排名累积来源概率质量的 log，初步得到每个排名的上尾。 |
| 52 | <code>    # Each tie receives the cumulative mass through its last member.</code> | ties 必须用整个等值块的尾质量，不能按出现顺序分配不同 p 值。 |
| 53 | <code>    end = torch.cat((ordered[:, 1:] != ordered[:, :-1],</code> | 标记每个 tie 块最后一个位置，词表末项强制为末尾。 |
| 54 | <code>                     torch.ones_like(ordered[:, :1], dtype=torch.bool)), dim=1)</code> | 标记每个 tie 块最后一个位置，词表末项强制为末尾。 |
| 55 | <code>    ranks = torch.arange(gap.shape[1], device=gap.device).expand_as(order)</code> | 构造每行0…V−1的排名下标供广播。 |
| 56 | <code>    ends = torch.where(end, ranks, gap.shape[1]-1).flip(-1).cummin(-1).values.flip(-1)</code> | 反向 cummin 找每个位置所属 tie 块的最后下标，保留整块质量。 |
| 57 | <code>    log_tail = log_tail.gather(1, ends).clamp_max(0)</code> | 将 tie 块末尾累计质量广播回块内，并裁 log tail≤0 抑制舍入。 |
| 58 | <code>    log_h = log_calibrator(log_tail)</code> | 尾质量经 log_calibrator 得 log h(u)。 |
| 59 | <code>    log_z = (sorted_s+log_h).logsumexp(-1)</code> | 计算 log E_source[h]，使 Q_rank 在完整词表上归一化。 |
| 60 | <code>    observed_rank = (order == targets[:, None]).long().argmax(-1)[:, None]</code> | 在排列 order 中找到实际目标 token 的排名，形状 [Q,1]。 |
| 61 | <code>    rank_gap = log_h.gather(1, observed_rank)[:, 0]-log_z</code> | 目标的 log h−log Z 即 Q_rank/source 的 log 比。 |
| 62 | <code>    tv = (.5*(log_s.exp()-log_0.exp()).abs().sum(-1)).clamp(0, 1)</code> | 两完整词表分布的总变差 TV=1/2Σ|p1−p0|，裁到[0,1]。 |
| 63 | <code>    gather = lambda array: array.gather(1, targets[:, None])[:, 0]</code> | 嵌套 lambda gather 捕获目标下标，从任何 [Q,V] 数组提取实际 token 的 [Q] 值。 |
| 64 | <code>    return {</code> | 以下返回八条逐预测位置数组。 |
| 65 | <code>        'calibrated_gap': torch.logaddexp(torch.log1p(-tv), tv.log()+rank_gap),</code> | 最终 log 比为 log[(1−TV)+TV*exp(rank_gap)]，TV=0 时回到零证据。 |
| 66 | <code>        'rank_gap': rank_gap,</code> | 暴露纯排名构造的 log 比。 |
| 67 | <code>        'source_tv': tv,</code> | 暴露 TV 门控强度，而不是标签置信度。 |
| 68 | <code>        'source_tail_logp': log_tail.gather(1, observed_rank)[:, 0],</code> | 暴露目标所在完整 tie 块的来源上尾 log 质量。 |
| 69 | <code>        'distribution_gap': gather(gap),</code> | 提取目标 token 的原始 FP64 absent/source log 比。 |
| 70 | <code>        'bridge_gap': .5*gather(gap)-(.5*(log_s+log_0)).logsumexp(-1),</code> | 几何桥分布相对来源的目标 log 比：.5*gap 减全词表几何均值归一化常数。 |
| 71 | <code>        'source_nll': -gather(log_s),</code> | 来源世界实际 token 的负 logp，即 surprisal。 |
| 72 | <code>        'source_entropy': -(log_s.exp()*log_s).sum(-1),</code> | 来源全词表 Shannon 熵，沿 V 轴归约；不是目标 token 的 NLL。 |
| 73 | <code>    }</code> | 关闭返回字典，所有值都保留 logits 设备。 |
| 74 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 75 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 76 | <code>class SourceDistributions:</code> | SourceDistributions 为一回答暂存 present logits，在 absent 到达时配对归约；由 Observer 在 conditional 时创建。 |
| 77 | <code>    """Hold one answer's source logits on CPU; reduce when absent rows arrive."""</code> | SourceDistributions 为一回答暂存 present logits，在 absent 到达时配对归约；由 Observer 在 conditional 时创建。 |
| 78 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 79 | <code>    def __init__(self, targets):</code> | 构造器输入原回答 IDs，保留长度和每块目标切片。 |
| 80 | <code>        self.targets = targets</code> | 保存 targets 引用；调用者应保持回答编码不变。 |
| 81 | <code>        self.source = {}</code> | 按块起点暂存 present logits，内存随尚未归约词表块累积。 |
| 82 | <code>        self.parts = {key: [] for key in DISTRIBUTION_FIELDS}</code> | 八个输出字段各自积累 CPU 数组块。 |
| 83 | <code>        self.present_rows = self.absent_rows = 0</code> | 两个世界已接受的行数都从0开始，用作严格调度游标。 |
| 84 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 85 | <code>    def present(self, start, logits):</code> | present(start,logits) 是 source_full 的回调，输入原生 logits 块。 |
| 86 | <code>        if start != self.present_rows or start+len(logits) &gt; len(self.targets):</code> | 块必须连续且不超过回答长度，拒绝重排/缺口。 |
| 87 | <code>            raise ValueError('Source distribution row schedule changed')</code> | 块必须连续且不超过回答长度，拒绝重排/缺口。 |
| 88 | <code>        self.source[start] = logits.detach().cpu()</code> | detach 后移 CPU 暂存，避免全部 present 词表占 GPU；不改变原 dtype。 |
| 89 | <code>        self.present_rows += len(logits)</code> | 递增 present 行数，不能重复接受同起点。 |
| 90 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 91 | <code>    def absent(self, start, logits):</code> | absent(start,logits) 配对同起点的 present 块，归约后释放暂存。 |
| 92 | <code>        if start != self.absent_rows or start not in self.source:</code> | absent 块连续且 present 已存在，否则两世界无法对齐。 |
| 93 | <code>            raise ValueError('Absent distribution row schedule changed')</code> | absent 块连续且 present 已存在，否则两世界无法对齐。 |
| 94 | <code>        source = self.source.pop(start)</code> | pop 取并移除配对块，减少 CPU logits 占用。 |
| 95 | <code>        if source.shape != logits.shape:</code> | 两世界块 [Q,V] 必须同形，否则目标/词表轴不一致。 |
| 96 | <code>            raise ValueError('Source/absent distribution chunks differ')</code> | 两世界块 [Q,V] 必须同形，否则目标/词表轴不一致。 |
| 97 | <code>        result = conditional_observations(source.to(logits.device), logits,</code> | present 移到 absent 设备后统一计算八项，目标使用原回答同一位置切片。 |
| 98 | <code>                                          self.targets[start:start+len(logits)])</code> | present 移到 absent 设备后统一计算八项，目标使用原回答同一位置切片。 |
| 99 | <code>        for key, value in result.items():</code> | 各项 detach/CPU/NumPy 后分块收集，不保留计算图。 |
| 100 | <code>            self.parts[key].append(value.detach().cpu().numpy())</code> | 各项 detach/CPU/NumPy 后分块收集，不保留计算图。 |
| 101 | <code>        self.absent_rows += len(logits)</code> | 归约成功才递增 absent 行数。 |
| 102 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 103 | <code>    def finish(self):</code> | finish 在两个世界观测完成后返回全回答诊断；由 Observer 调用。 |
| 104 | <code>        if self.source or self.present_rows != self.absent_rows or self.absent_rows != len(self.targets):</code> | 必须无剩余 present 块且两游标都等于回答长度；未配对/漏块报错。 |
| 105 | <code>            raise ValueError('Incomplete source/absent distribution observation')</code> | 必须无剩余 present 块且两游标都等于回答长度；未配对/漏块报错。 |
| 106 | <code>        return {key: np.concatenate(parts) for key, parts in self.parts.items()}</code> | 按接受顺序拼接每项 [Q] 块成为 [N]；公共入口保证非空回答。 |
