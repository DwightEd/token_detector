# myproject/segmentation.py 逐行讲解

基线：`1210120748b3a05bd271bc79debcbed4fdc0c964`；SHA-256：`300032e312aefefbc9f1a3c9006c5c78024f9f14517f52ad17f047bec69fd62c`；136 行。

源码与注释逐行并排。一个跨行表达式共用同一操作说明；这不代表各行独立执行。完整调用路径、输入输出轴和理论边界见 [METHOD.md](../METHOD.md)。未执行的 GPU 路径见 [验证边界](../VALIDATION.md)。

| 函数（含嵌套） | 原始行号 | 作用与契约入口 |
|---|---|---|
| `partition_scores` | 8–35 | partition_scores 接收 [N+1,N+1] 区间收益、合法切点及非负切分成本，返回最优 cuts 和目标。由 segment/predictive_segment 调用。 |
| `partition_scores.<lambda>@29:31` | 29–29 | 嵌套 lambda 比较 (已有段数,前端点)，先偏好更少片段、再更小起点，结果确定。 |
| `legal_atoms` | 37–48 | legal_atoms 从原文本与 token offsets 得可切端点及 bool mask；由 API.measure 调用，不能重分词。 |
| `_inputs` | 51–59 | _inputs 对 [N,D] 概率剖面与同形因果可用轴做校验，供区间与拟合共用。 |
| `background` | 61–64 | background 返回共享目的地先验[D]，用一个总质量为1的均匀事件平滑全部行累计。 |
| `_prediction` | 66–68 | _prediction 接收过去行累计 counts、背景 base 和当前行可用 mask，返回当前预序列预测[D]。 |
| `interval_gains` | 70–85 | interval_gains 返回所有 [first,last) 的预序列期望 log-score 相对背景增益矩阵；用于选分段，不是事实正确性得分。 |
| `fitted_profiles` | 87–104 | fitted_profiles 给定 cuts，输出逐 token 预测/残差、每片段共享状态和背景；保留所有个体残差。 |
| `segment` | 106–111 | segment 是无标签单视图入口，返回 cuts/目标/惩罚/区间表及拟合诊断。 |
| `predictive_gain` | 113–120 | predictive_gain 用 fit 的状态预测 hold 视图，返回每 token 平均增益；不在 hold 重新估计参数。 |
| `predictive_segment` | 123–136 | predictive_segment 在粗惩罚/零惩罚两种分段间按第二 head 视图选；两视图相关且 hold 被用于选择，不能称 untouched validation。 |

| 行 | 原始代码 | 操作、目的与边界 |
|---|---|---|
| 1 | <code>"""Deterministic directed dependency states and word-external contiguous partitions."""</code> | 无标签依赖分段：有向预测剖面状态与词外连续切点，非默认 punct 的可选路径。 |
| 2 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 3 | <code>import re</code> | 正则识别词及撇号连写，限制不能在词内部切 token。 |
| 4 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 5 | <code>import numpy as np</code> | NumPy 实现区间动态规划、因果 mask 和概率剖面。 |
| 6 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 7 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 8 | <code>def partition_scores(scores, atoms=None, cut_penalty=0.0):</code> | partition_scores 接收 [N+1,N+1] 区间收益、合法切点及非负切分成本，返回最优 cuts 和目标。由 segment/predictive_segment 调用。 |
| 9 | <code>    """Optimize arbitrary interval scores on legal endpoints; charge internal cuts."""</code> | partition_scores 接收 [N+1,N+1] 区间收益、合法切点及非负切分成本，返回最优 cuts 和目标。由 segment/predictive_segment 调用。 |
| 10 | <code>    scores = np.asarray(scores, dtype=float)</code> | 区间收益转浮点，−Inf 表示不可用区间。 |
| 11 | <code>    if scores.ndim != 2 or scores.shape[0] != scores.shape[1] or len(scores) &lt; 2:</code> | 必须方阵且至少一个 token，才有从0到N的区间。 |
| 12 | <code>        raise ValueError('Expected an (n+1)-square interval score matrix')</code> | 必须方阵且至少一个 token，才有从0到N的区间。 |
| 13 | <code>    n = len(scores) - 1</code> | 端点总数减1得到原始 token 数。 |
| 14 | <code>    atoms = np.arange(n + 1) if atoms is None else np.asarray(atoms)</code> | 缺省允许所有端点；否则保留调用者给定合法 atoms。 |
| 15 | <code>    if (atoms.ndim != 1 or len(atoms) &lt; 2 or atoms.dtype.kind not in 'iu'</code> | 要求整型、至少首尾两点、严格递增、覆盖0和N，切分惩罚有限非负；不符合则拒绝。 |
| 16 | <code>            or atoms[0] != 0 or atoms[-1] != n or np.any(np.diff(atoms) &lt;= 0)</code> | 要求整型、至少首尾两点、严格递增、覆盖0和N，切分惩罚有限非负；不符合则拒绝。 |
| 17 | <code>            or not np.isfinite(cut_penalty) or cut_penalty &lt; 0):</code> | 要求整型、至少首尾两点、严格递增、覆盖0和N，切分惩罚有限非负；不符合则拒绝。 |
| 18 | <code>        raise ValueError('Expected complete increasing endpoints and nonnegative cut cost')</code> | 要求整型、至少首尾两点、严格递增、覆盖0和N，切分惩罚有限非负；不符合则拒绝。 |
| 19 | <code>    best = np.full(n + 1, -np.inf)</code> | best[b] 是到端点 b 的最大累计收益，初始不可达。 |
| 20 | <code>    count = np.zeros(n + 1, dtype=int)</code> | count[b] 记录达到 b 使用的片段数，作确定性 tie-break。 |
| 21 | <code>    parent = np.zeros(n + 1, dtype=int)</code> | parent[b] 记录最优上一个切点用于回溯。 |
| 22 | <code>    best[0] = 0.0</code> | 空前缀目标为0。 |
| 23 | <code>    for index, b in enumerate(atoms[1:], 1):</code> | 对每个合法右端 b，只枚举更早的合法起点，形成有向无环 DP。 |
| 24 | <code>        previous = atoms[:index]</code> | 对每个合法右端 b，只枚举更早的合法起点，形成有向无环 DP。 |
| 25 | <code>        candidate = best[previous] + scores[previous, b] - cut_penalty * (previous &gt; 0)</code> | 候选=前缀最优+区间收益−内部切点惩罚；起点0不收费。 |
| 26 | <code>        if not np.isfinite(candidate).any():</code> | 当前合法端点没有有限可达方案就报错，不返回不完整分段。 |
| 27 | <code>            raise ValueError('No finite partition reaches an allowed endpoint')</code> | 当前合法端点没有有限可达方案就报错，不返回不完整分段。 |
| 28 | <code>        tied = np.flatnonzero(candidate &gt;= candidate.max() - 1e-12)</code> | 在1e−12容差内视作收益 ties，是明确数值选择规则。 |
| 29 | <code>        chosen = min(tied, key=lambda k: (count[previous[k]], previous[k]))</code> | 嵌套 lambda 比较 (已有段数,前端点)，先偏好更少片段、再更小起点，结果确定。 |
| 30 | <code>        a = previous[chosen]</code> | 将选中候选的局部序号映回真实端点 a。 |
| 31 | <code>        (best[b], count[b], parent[b]) = (candidate[chosen], count[a] + 1, a)</code> | 同时更新 b 的目标、段数、前驱，供之后右端使用。 |
| 32 | <code>    cuts = [n]</code> | 从最终端点N开始回溯。 |
| 33 | <code>    while cuts[-1] &gt; 0:</code> | 沿严格更小 parent 返回0，有限终止；反向 cuts 还不是输出顺序。 |
| 34 | <code>        cuts.append(int(parent[cuts[-1]]))</code> | 沿严格更小 parent 返回0，有限终止；反向 cuts 还不是输出顺序。 |
| 35 | <code>    return (np.array(cuts[::-1]), float(best[n]))</code> | 反转得到0…N递增切点，返回最终目标浮点。 |
| 36 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 37 | <code>def legal_atoms(response):</code> | legal_atoms 从原文本与 token offsets 得可切端点及 bool mask；由 API.measure 调用，不能重分词。 |
| 38 | <code>    """Allow word-external, nonoverlapping cuts on the original token axis."""</code> | legal_atoms 从原文本与 token offsets 得可切端点及 bool mask；由 API.measure 调用，不能重分词。 |
| 39 | <code>    text = response['text']</code> | 保留原文本与 [N,2] offset 矩阵。 |
| 40 | <code>    offsets = np.array(response['offsets'])</code> | 保留原文本与 [N,2] offset 矩阵。 |
| 41 | <code>    interior = np.zeros(len(text) + 1, dtype=bool)</code> | interior 按字符边界建立是否位于词内的表。 |
| 42 | <code>    for word in re.finditer("\\w+(?:['’]\\w+)*", text):</code> | Unicode \w 及词内撇号视为同词，只把首尾之间边界标为词内；这不是通用语义 parser。 |
| 43 | <code>        interior[word.start() + 1:word.end()] = True</code> | Unicode \w 及词内撇号视为同词，只把首尾之间边界标为词内；这不是通用语义 parser。 |
| 44 | <code>    allowed = np.ones(len(offsets) + 1, dtype=bool)</code> | 默认首尾及中间 token 切点都允许。 |
| 45 | <code>    for cut in range(1, len(offsets)):</code> | 逐内部切点取右 token 的原文字符起点。 |
| 46 | <code>        position = int(offsets[cut, 0])</code> | 逐内部切点取右 token 的原文字符起点。 |
| 47 | <code>        allowed[cut] = not (offsets[cut - 1, 1] &gt; position or interior[position])</code> | 若左 token offset 与右 token 重叠，或字符边界在词内，则禁止该切点。 |
| 48 | <code>    return (np.flatnonzero(allowed), allowed)</code> | 返回合法 token 端点索引和完整 mask，便于后续 DP 约束。 |
| 49 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 50 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 51 | <code>def _inputs(profiles, available):</code> | _inputs 对 [N,D] 概率剖面与同形因果可用轴做校验，供区间与拟合共用。 |
| 52 | <code>    p = np.asarray(profiles, dtype=np.float64)</code> | 概率转 float64，available 转 bool。 |
| 53 | <code>    mask = np.asarray(available, dtype=bool)</code> | 概率转 float64，available 转 bool。 |
| 54 | <code>    if (p.ndim != 2 or not len(p) or p.shape != mask.shape or not p.shape[1]</code> | 要求非空二维、同形、有限非负、不可用位置质量为0、每行和近似1，否则拒绝。 |
| 55 | <code>            or not np.isfinite(p).all() or np.any(p &lt; 0) or np.any(p[~mask] != 0)</code> | 要求非空二维、同形、有限非负、不可用位置质量为0、每行和近似1，否则拒绝。 |
| 56 | <code>            or not np.allclose(p.sum(1), 1, atol=1e-5, rtol=0)):</code> | 要求非空二维、同形、有限非负、不可用位置质量为0、每行和近似1，否则拒绝。 |
| 57 | <code>        raise ValueError('Expected normalized nonnegative routing on its causal support')</code> | 要求非空二维、同形、有限非负、不可用位置质量为0、每行和近似1，否则拒绝。 |
| 58 | <code>    p = p / p.sum(1, keepdims=True)</code> | 逐行精确重归一，消除 attention 舍入导致的小质量偏移。 |
| 59 | <code>    return p, mask</code> | 返回归一化剖面与原因果支持 mask。 |
| 60 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 61 | <code>def background(profiles):</code> | background 返回共享目的地先验[D]，用一个总质量为1的均匀事件平滑全部行累计。 |
| 62 | <code>    """One total uniform routing event smooths the empirical destination prior."""</code> | background 返回共享目的地先验[D]，用一个总质量为1的均匀事件平滑全部行累计。 |
| 63 | <code>    n, d = profiles.shape</code> | n 是 token 行数，d 是可路由目的地总坐标数。 |
| 64 | <code>    return (profiles.sum(0) + 1 / d) / (n + 1)</code> | 每列加1/d，再除 n+1，形成正、归一化的背景概率。 |
| 65 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 66 | <code>def _prediction(counts, base, mask):</code> | _prediction 接收过去行累计 counts、背景 base 和当前行可用 mask，返回当前预序列预测[D]。 |
| 67 | <code>    q = np.where(mask, base + counts, 0.)</code> | 只在当前因果可用目的地保留背景加历史计数。 |
| 68 | <code>    return q / q.sum()</code> | 在当前可用支持重新归一，禁止未来 key 泄露进当前预测。 |
| 69 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 70 | <code>def interval_gains(profiles, available):</code> | interval_gains 返回所有 [first,last) 的预序列期望 log-score 相对背景增益矩阵；用于选分段，不是事实正确性得分。 |
| 71 | <code>    """Expected prequential log-score gain over the common restricted background."""</code> | interval_gains 返回所有 [first,last) 的预序列期望 log-score 相对背景增益矩阵；用于选分段，不是事实正确性得分。 |
| 72 | <code>    p, mask = _inputs(profiles, available)</code> | 校验归一化输入并构造共享背景。 |
| 73 | <code>    base = background(p)</code> | 校验归一化输入并构造共享背景。 |
| 74 | <code>    baseline = base[None, :] * mask</code> | 将背景限制到每行因果支持，再逐行归一作为比较基准。 |
| 75 | <code>    baseline /= baseline.sum(1, keepdims=True)</code> | 将背景限制到每行因果支持，再逐行归一作为比较基准。 |
| 76 | <code>    scores = np.full((len(p) + 1, len(p) + 1), -np.inf)</code> | 分配 [N+1,N+1] 区间收益表，非法/未计算区间 −Inf。 |
| 77 | <code>    for first in range(len(p)):</code> | 逐候选起点独立重置历史 counts[D] 与累计 gain。 |
| 78 | <code>        counts, gain = np.zeros(p.shape[1]), 0.</code> | 逐候选起点独立重置历史 counts[D] 与累计 gain。 |
| 79 | <code>        for last in range(first, len(p)):</code> | 逐步延伸右端，对每个可达区间只增量加入一行。 |
| 80 | <code>            q = _prediction(counts, base, mask[last])</code> | 当前预测只用此前已累计行，不先看当前剖面。 |
| 81 | <code>            active = p[last] &gt; 0</code> | 只在当前 p>0 位置计算 log 比，避免0*log0。 |
| 82 | <code>            gain += p[last, active] @ (np.log(q[active]) - np.log(baseline[last, active]))</code> | 用当前概率剖面对预测/背景 log 比取期望，并累加区间收益。 |
| 83 | <code>            scores[first, last + 1] = gain</code> | 将包括当前行的区间 [first,last+1) 收益存表。 |
| 84 | <code>            counts += p[last]</code> | 评分后才把当前剖面加入历史，保留 prequential 顺序。 |
| 85 | <code>    return scores</code> | 返回区间表；复杂度 O(N²D)，不是线性时间。 |
| 86 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 87 | <code>def fitted_profiles(profiles, available, cuts):</code> | fitted_profiles 给定 cuts，输出逐 token 预测/残差、每片段共享状态和背景；保留所有个体残差。 |
| 88 | <code>    """Keep shared counts, token-specific predictions and all individual residuals."""</code> | fitted_profiles 给定 cuts，输出逐 token 预测/残差、每片段共享状态和背景；保留所有个体残差。 |
| 89 | <code>    p, mask = _inputs(profiles, available)</code> | 验证剖面，转 cuts 数组。 |
| 90 | <code>    cuts = np.asarray(cuts)</code> | 验证剖面，转 cuts 数组。 |
| 91 | <code>    if (cuts.ndim != 1 or cuts.dtype.kind not in 'iu' or cuts[0] != 0</code> | 切点必须整数、始0终N严格递增；极端空 cuts 会先索引失败，当前未专门兜底。 |
| 92 | <code>            or cuts[-1] != len(p) or np.any(np.diff(cuts) &lt;= 0)):</code> | 切点必须整数、始0终N严格递增；极端空 cuts 会先索引失败，当前未专门兜底。 |
| 93 | <code>        raise ValueError('Cuts must partition the original rows')</code> | 切点必须整数、始0终N严格递增；极端空 cuts 会先索引失败，当前未专门兜底。 |
| 94 | <code>    base = background(p)</code> | 同一个回答背景供所有片段使用。 |
| 95 | <code>    predicted = np.zeros_like(p)</code> | 分配 [N,D] token 预测和片段状态列表。 |
| 96 | <code>    states = []</code> | 分配 [N,D] token 预测和片段状态列表。 |
| 97 | <code>    for first, last in zip(cuts[:-1], cuts[1:]):</code> | 每片段单独清零目的地计数，禁止状态跨段。 |
| 98 | <code>        counts = np.zeros(p.shape[1])</code> | 每片段单独清零目的地计数，禁止状态跨段。 |
| 99 | <code>        for row in range(first, last):</code> | 逐 token 先以此前计数预测，再加入当前剖面；predicted 仍是 token 条件分布。 |
| 100 | <code>            predicted[row] = _prediction(counts, base, mask[row])</code> | 逐 token 先以此前计数预测，再加入当前剖面；predicted 仍是 token 条件分布。 |
| 101 | <code>            counts += p[row]</code> | 逐 token 先以此前计数预测，再加入当前剖面；predicted 仍是 token 条件分布。 |
| 102 | <code>        states.append((base + counts) / (1 + last - first))</code> | 段末以背景+全段计数形成共享状态，分母是1+段长。 |
| 103 | <code>    return {'predicted': predicted, 'residual': p - predicted,</code> | 返回个体预测与 p−predicted 残差，以及 [段数,D] shared 和[D]背景。 |
| 104 | <code>            'shared': np.asarray(states), 'background': base}</code> | 返回个体预测与 p−predicted 残差，以及 [段数,D] shared 和[D]背景。 |
| 105 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 106 | <code>def segment(profiles, available, atoms):</code> | segment 是无标签单视图入口，返回 cuts/目标/惩罚/区间表及拟合诊断。 |
| 107 | <code>    scores = interval_gains(profiles, available)</code> | 从 fit attention 计算全部区间增益。 |
| 108 | <code>    penalty = float(np.log(2 * max(len(atoms) - 2, 1)))</code> | 固定切点惩罚 log(2*max(内部合法切点数,1))，不是用错误标签调出的。 |
| 109 | <code>    cuts, objective = partition_scores(scores, atoms, penalty)</code> | 在合法 atoms 上运行最优区间 DP。 |
| 110 | <code>    return {'cuts': cuts, 'objective': objective, 'cut_penalty': penalty, 'interval_gains': scores,</code> | 附 DP 与 token 预测诊断，供 API 只取 cuts 或研究检查。 |
| 111 | <code>            **fitted_profiles(profiles, available, cuts)}</code> | 附 DP 与 token 预测诊断，供 API 只取 cuts 或研究检查。 |
| 112 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 113 | <code>def predictive_gain(fit, hold, available, cuts):</code> | predictive_gain 用 fit 的状态预测 hold 视图，返回每 token 平均增益；不在 hold 重新估计参数。 |
| 114 | <code>    """Transfer fit-view predictive states; no held-view parameter estimation."""</code> | predictive_gain 用 fit 的状态预测 hold 视图，返回每 token 平均增益；不在 hold 重新估计参数。 |
| 115 | <code>    fitted = fitted_profiles(fit, available, cuts)</code> | 按给定 cuts 拟合 fit 视图预测。 |
| 116 | <code>    p, mask = _inputs(hold, available)</code> | 独立检查 hold 视图的概率/因果支持。 |
| 117 | <code>    base = fitted['background'][None, :] * mask</code> | 使用 fit 背景并在 hold 因果支持上归一作基准。 |
| 118 | <code>    base /= base.sum(1, keepdims=True)</code> | 使用 fit 背景并在 hold 因果支持上归一作基准。 |
| 119 | <code>    active = p &gt; 0</code> | 只在 hold 的正概率位置评价 log 比。 |
| 120 | <code>    return float(np.sum(p[active] * (np.log(fitted['predicted'][active]) - np.log(base[active]))) / len(p))</code> | 按 hold 概率加权，汇总全部位置后除 token 数；不是独立测试误差。 |
| 121 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 122 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 123 | <code>def predictive_segment(fit, hold, available, atoms):</code> | predictive_segment 在粗惩罚/零惩罚两种分段间按第二 head 视图选；两视图相关且 hold 被用于选择，不能称 untouched validation。 |
| 124 | <code>    """Choose coarse/zero-cost partitions by routing prediction, without error labels.</code> | predictive_segment 在粗惩罚/零惩罚两种分段间按第二 head 视图选；两视图相关且 hold 被用于选择，不能称 untouched validation。 |
| 125 | <code> </code> | predictive_segment 在粗惩罚/零惩罚两种分段间按第二 head 视图选；两视图相关且 hold 被用于选择，不能称 untouched validation。 |
| 126 | <code>    The head views are correlated and the second is used for selection; its gain</code> | predictive_segment 在粗惩罚/零惩罚两种分段间按第二 head 视图选；两视图相关且 hold 被用于选择，不能称 untouched validation。 |
| 127 | <code>    is consequently not an untouched validation estimate or a correctness score.</code> | predictive_segment 在粗惩罚/零惩罚两种分段间按第二 head 视图选；两视图相关且 hold 被用于选择，不能称 untouched validation。 |
| 128 | <code>    """</code> | predictive_segment 在粗惩罚/零惩罚两种分段间按第二 head 视图选；两视图相关且 hold 被用于选择，不能称 untouched validation。 |
| 129 | <code>    scores = interval_gains(fit, available)</code> | 区间收益只在 fit 视图计算。 |
| 130 | <code>    penalties = [float(np.log(2 * max(len(atoms) - 2, 1))), 0.]</code> | 仅两个固定候选惩罚，不搜索测试标签。 |
| 131 | <code>    candidates = [partition_scores(scores, atoms, penalty)[0] for penalty in penalties]</code> | 分别运行同一 DP 得两个候选 cuts。 |
| 132 | <code>    gains = [predictive_gain(fit, hold, available, cuts) for cuts in candidates]</code> | 将 fit 预测转移到 hold 计算两种 gain。 |
| 133 | <code>    chosen = int(gains[1] &gt; gains[0])</code> | 只有零惩罚 gain 严格更高才选1，ties保留粗惩罚。 |
| 134 | <code>    return {'cuts': candidates[chosen], 'cut_penalty': penalties[chosen],</code> | 返回所选 cuts/惩罚、两个 selection gains 与 fit 视图完整诊断。 |
| 135 | <code>            'selection_gains': np.asarray(gains),</code> | 返回所选 cuts/惩罚、两个 selection gains 与 fit 视图完整诊断。 |
| 136 | <code>            **fitted_profiles(fit, available, candidates[chosen])}</code> | 返回所选 cuts/惩罚、两个 selection gains 与 fit 视图完整诊断。 |
