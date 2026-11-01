# myproject/revision.py 逐行讲解

基线：`1210120748b3a05bd271bc79debcbed4fdc0c964`；SHA-256：`07b146366085dbb06dd27ce8406918f78035a666b6a0742ca4b84a26cc50890e`；211 行。

源码与注释逐行并排。一个跨行表达式共用同一操作说明；这不代表各行独立执行。完整调用路径、输入输出轴和理论边界见 [METHOD.md](../METHOD.md)。未执行的 GPU 路径见 [验证边界](../VALIDATION.md)。

| 函数（含嵌套） | 原始行号 | 作用与契约入口 |
|---|---|---|
| `proof_pack` | 17–21 | proof_pack输入原pack、来源、回答、offset与可选question；返回附proofs的浅拷贝，不改已有数值字段。由API.predict的显式proof_revision分支调用。 |
| `_or_destination` | 24–27 | _or_destination(size)缓存两个mask做OR后的展平目的地[size²]，供块间事件卷积。 |
| `_convolve` | 30–33 | _convolve接收两个同大小的log mask分布，返回OR合并后的分布；由跨块prefix/suffix调用。 |
| `bounded_event_probability` | 36–60 | bounded_event_probability给定基础事件log p与预算B，返回最大可行事件概率a及混合权重w；基于D_KL(new||base)的Bernoulli约化，不评价proof真伪。 |
| `bounded_event_probability.divergence` | 55–57 | 嵌套divergence捕获log p、log(1−p)、budget，输入候选事件概率a，返回KL−预算供根求解。 |
| `_MaskedBlock.__init__` | 66–84 | 构造器输入片段d、每token角色bitmask和size=2^角色数；输出状态保存在实例供marginals复用。 |
| `_MaskedBlock.marginals` | 86–103 | marginals(log_accept)接收其他块传来的每mask接受log因子，返回当前片段条件事件下的token state1概率。 |
| `_canonical_component` | 106–120 | _canonical_component接收编译器一个不兼容DNF组件及有效token数，返回去冗余的角色token集合之合取项。 |
| `revise_scores` | 123–211 | revise_scores 输入原d及回答/片段轴、DNF组件、KL与规模预算，返回bounded/hard/witness及诊断。proof相连片段按关联分量共享一个预算，不跨分量连块。 |

| 行 | 原始代码 | 操作、目的与边界 |
|---|---|---|
| 1 | <code>"""Exact source-event inference with a fixed information-revision budget.</code> | 可选proof_revision的精确事件条件推断与KL修改预算；限制相对V3的变化，不能验证证明本身正确，更不是校准事实概率。 |
| 2 | <code> </code> | 可选proof_revision的精确事件条件推断与KL修改预算；限制相对V3的变化，不能验证证明本身正确，更不是校准事实概率。 |
| 3 | <code>These are compatibility-conditioned diagnostics, not calibrated error probabilities.</code> | 可选proof_revision的精确事件条件推断与KL修改预算；限制相对V3的变化，不能验证证明本身正确，更不是校准事实概率。 |
| 4 | <code>The budget limits change from V3; it does not estimate whether a proof is correct.</code> | 可选proof_revision的精确事件条件推断与KL修改预算；限制相对V3的变化，不能验证证明本身正确，更不是校准事实概率。 |
| 5 | <code>"""</code> | 可选proof_revision的精确事件条件推断与KL修改预算；限制相对V3的变化，不能验证证明本身正确，更不是校准事实概率。 |
| 6 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 7 | <code>from functools import lru_cache</code> | lru_cache 缓存按mask数决定的OR目的地索引，最多16种大小。 |
| 8 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 9 | <code>import numpy as np</code> | NumPy处理bitmask和对数联合质量。 |
| 10 | <code>from scipy.optimize import brentq</code> | brentq在[p,1]上求满足Bernoulli KL预算的根；不是训练梯度下降。 |
| 11 | <code>from scipy.special import expit, logsumexp, xlogy</code> | expit把V3 odds变概率；logsumexp边缘化；xlogy稳定处理0log0。 |
| 12 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 13 | <code>from .detector import unit_ranges</code> | 复用原始回答/片段连续边界。 |
| 14 | <code>from .sequence import _grid, grouped_persistent_scores, witness_emissions</code> | 复用同一个V3求积、分组内核与witness发射，不另外定义基础检测器。 |
| 15 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 16 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 17 | <code>def proof_pack(pack, source, response, offsets, question=None):</code> | proof_pack输入原pack、来源、回答、offset与可选question；返回附proofs的浅拷贝，不改已有数值字段。由API.predict的显式proof_revision分支调用。 |
| 18 | <code>    """Attach source-only proof metadata without changing native numerical fields."""</code> | proof_pack输入原pack、来源、回答、offset与可选question；返回附proofs的浅拷贝，不改已有数值字段。由API.predict的显式proof_revision分支调用。 |
| 19 | <code>    from .proofs import compile_proofs</code> | 延迟引入有限语法编译器，不在普通推断加载额外规则。 |
| 20 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 21 | <code>    return {**pack, 'proofs': compile_proofs(source, response, offsets, pack['target'], question)}</code> | 将原target轴传给compile_proofs，proof的token索引因此指向有效评分轴而非原offset全轴。 |
| 22 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 23 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 24 | <code>@lru_cache(maxsize=16)</code> | _or_destination(size)缓存两个mask做OR后的展平目的地[size²]，供块间事件卷积。 |
| 25 | <code>def _or_destination(size):</code> | _or_destination(size)缓存两个mask做OR后的展平目的地[size²]，供块间事件卷积。 |
| 26 | <code>    masks = np.arange(size)</code> | 所有可能role命中状态为0…size−1，size通常2^角色数。 |
| 27 | <code>    return (masks[:, None] &#124; masks[None, :]).ravel()</code> | 通过广播枚举每个mask对的按位OR，形状[size,size]再展平。 |
| 28 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 29 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 30 | <code>def _convolve(left, right):</code> | _convolve接收两个同大小的log mask分布，返回OR合并后的分布；由跨块prefix/suffix调用。 |
| 31 | <code>    result = np.full(len(left), -np.inf)</code> | 输出各OR状态初始不可达−Inf。 |
| 32 | <code>    np.logaddexp.at(result, _or_destination(len(left)), (left[:, None]+right).ravel())</code> | 两独立块联合log质量相加，并用logaddexp.at累加所有落到同一OR mask的路径，不能普通覆盖赋值。 |
| 33 | <code>    return result</code> | 返回size项mask分布；时间O(size²)，随role数指数增长。 |
| 34 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 35 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 36 | <code>def bounded_event_probability(log_probability, budget=np.log(2.)):</code> | bounded_event_probability给定基础事件log p与预算B，返回最大可行事件概率a及混合权重w；基于D_KL(new||base)的Bernoulli约化，不评价proof真伪。 |
| 37 | <code>    """Maximize event probability within forward KL, then minimize KL on ties.</code> | bounded_event_probability给定基础事件log p与预算B，返回最大可行事件概率a及混合权重w；基于D_KL(new||base)的Bernoulli约化，不评价proof真伪。 |
| 38 | <code> </code> | bounded_event_probability给定基础事件log p与预算B，返回最大可行事件概率a及混合权重w；基于D_KL(new||base)的Bernoulli约化，不评价proof真伪。 |
| 39 | <code>    A finite log probability avoids underflow for rare events. Returns the</code> | bounded_event_probability给定基础事件log p与预算B，返回最大可行事件概率a及混合权重w；基于D_KL(new||base)的Bernoulli约化，不评价proof真伪。 |
| 40 | <code>    event probability and weight of the hard conditional in a mixture with V3.</code> | bounded_event_probability给定基础事件log p与预算B，返回最大可行事件概率a及混合权重w；基于D_KL(new||base)的Bernoulli约化，不评价proof真伪。 |
| 41 | <code>    """</code> | bounded_event_probability给定基础事件log p与预算B，返回最大可行事件概率a及混合权重w；基于D_KL(new||base)的Bernoulli约化，不评价proof真伪。 |
| 42 | <code>    if (not np.isfinite(log_probability) or log_probability &gt; 1e-10</code> | 要求有限log p≤0（容忍1e−10舍入）及非负有限预算；p=0的−Inf不在此接口支持范围。 |
| 43 | <code>            or not np.isfinite(budget) or budget &lt; 0):</code> | 要求有限log p≤0（容忍1e−10舍入）及非负有限预算；p=0的−Inf不在此接口支持范围。 |
| 44 | <code>        raise ValueError('Finite log event probability &lt;=0 and nonnegative budget required')</code> | 要求有限log p≤0（容忍1e−10舍入）及非负有限预算；p=0的−Inf不在此接口支持范围。 |
| 45 | <code>    log_probability = min(float(log_probability), 0.)</code> | 把容忍的微小正log p裁回0。 |
| 46 | <code>    p = float(np.exp(log_probability))</code> | 计算p；极小p可能下溢为0，但后续KL仍用原log p保持罕见事件信息。 |
| 47 | <code>    if p == 1.:</code> | p=1时已确定事件，返回1与零修改权重。 |
| 48 | <code>        return 1., 0.</code> | p=1时已确定事件，返回1与零修改权重。 |
| 49 | <code>    if budget == 0:</code> | 零预算维持原p，不改变基础分布。 |
| 50 | <code>        return p, 0.</code> | 零预算维持原p，不改变基础分布。 |
| 51 | <code>    if -log_probability &lt;= budget:</code> | 完全条件化到事件的KL代价是−log p；预算足够时返回a=1、w=1。 |
| 52 | <code>        return 1., 1.</code> | 完全条件化到事件的KL代价是−log p；预算足够时返回a=1、w=1。 |
| 53 | <code>    log_other = float(np.log1p(-p))</code> | 稳定计算log(1−p)，作为非事件分区基准质量。 |
| 54 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 55 | <code>    def divergence(a):</code> | 嵌套divergence捕获log p、log(1−p)、budget，输入候选事件概率a，返回KL−预算供根求解。 |
| 56 | <code>        return (xlogy(a, a)-a*log_probability+xlogy(1-a, 1-a)</code> | 精确二元KL a log(a/p)+(1−a)log((1−a)/(1−p))−B，xlogy使端点0log0有定义。 |
| 57 | <code>                -(1-a)*log_other-budget)</code> | 精确二元KL a log(a/p)+(1−a)log((1−a)/(1−p))−B，xlogy使端点0log0有定义。 |
| 58 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 59 | <code>    a = float(brentq(divergence, p, 1., xtol=1e-14, rtol=1e-14))</code> | 在p与1之间求唯一上侧预算边界，绝对/相对容差均1e−14；求根失败向上传播。 |
| 60 | <code>    return a, (a-p)/(1-p)</code> | w=(a−p)/(1−p)，使(1−w)P+wP(.|A)的事件质量为a；这个分布保持各分区内部条件律。 |
| 61 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 62 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 63 | <code>class _MaskedBlock:</code> | _MaskedBlock为一个原始V3片段增加role命中OR位；被revise_scores按proof连接的块创建，持有前向表和块mask分布。 |
| 64 | <code>    """One original V3 block, extended by OR-hit bits for typed roles."""</code> | _MaskedBlock为一个原始V3片段增加role命中OR位；被revise_scores按proof连接的块创建，持有前向表和块mask分布。 |
| 65 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 66 | <code>    def __init__(self, values, membership, size):</code> | 构造器输入片段d、每token角色bitmask和size=2^角色数；输出状态保存在实例供marginals复用。 |
| 67 | <code>        self.e = witness_emissions(values)</code> | 基础发射固定witness，不把proof当新的LM概率。 |
| 68 | <code>        self.membership = np.asarray(membership, int)</code> | membership[N]表示每token命中哪些角色，转整数bitmask。 |
| 69 | <code>        pi, _, self.weights, self.transition = _grid(len(values), True)</code> | 复用uniform π,c网格与log权重/转移，沿用V3模型。 |
| 70 | <code>        self.masks = np.arange(size)</code> | masks枚举role事件状态；n是片段长度，g是求积节点数。 |
| 71 | <code>        n, g = len(values), len(pi)</code> | masks枚举role事件状态；n是片段长度，g是求积节点数。 |
| 72 | <code>        self.forward = np.full((n, g, size, 2), -np.inf)</code> | 前向表[n,g,size,2]，额外mask轴使内存相对V3再乘2^R。 |
| 73 | <code>        self.forward[0, :, 0, 0] = np.log1p(-pi)</code> | 初始state0没有role命中，质量log(1−π)放mask0。 |
| 74 | <code>        self.forward[0, :, self.membership[0], 1] = np.log(pi)+self.e[0]</code> | 初始state1命中首token角色，质量logπ+发射放对应mask。 |
| 75 | <code>        t00, t01, t10, t11 = (a[:, None] for a in self.transition)</code> | 四条转移改为[g,1]以沿role-mask轴广播。 |
| 76 | <code>        rows = np.arange(g)[:, None]</code> | 节点行索引[g,1]供np.logaddexp.at广播scatter使用。 |
| 77 | <code>        for t in range(1, n):</code> | 按后续token前推，previous/current是[g,size,2]视图。 |
| 78 | <code>            previous, current = self.forward[t-1], self.forward[t]</code> | 按后续token前推，previous/current是[g,size,2]视图。 |
| 79 | <code>            current[:, :, 0] = np.logaddexp(previous[:, :, 0]+t00, previous[:, :, 1]+t10)</code> | 到state0不命中新角色，旧mask不变，累加00和10路径。 |
| 80 | <code>            one = self.e[t]+np.logaddexp(previous[:, :, 0]+t01, previous[:, :, 1]+t11)</code> | 到state1计算01/11联合质量并加当前witness发射。 |
| 81 | <code>            destination = self.masks &#124; self.membership[t]</code> | state1使已命中mask与当前token角色做OR，多旧mask可能合流。 |
| 82 | <code>            np.logaddexp.at(current[:, :, 1], (rows, destination[None, :]), one)</code> | 用at稳定scatter-add所有合流路径，避免重复目的地被覆盖。 |
| 83 | <code>        terminal = logsumexp(self.forward[-1]+self.weights[:, None, None], axis=(0, 2))</code> | 片段末端对求积节点与0/1状态边缘化，得到每mask的未归一化质量。 |
| 84 | <code>        self.log_table = terminal-logsumexp(terminal)</code> | 对mask归一化成片段role命中分布，供不同块卷积；它不是事实错误分布。 |
| 85 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 86 | <code>    def marginals(self, log_accept):</code> | marginals(log_accept)接收其他块传来的每mask接受log因子，返回当前片段条件事件下的token state1概率。 |
| 87 | <code>        """Normalize integrated joints once, after the external circuit message."""</code> | marginals(log_accept)接收其他块传来的每mask接受log因子，返回当前片段条件事件下的token state1概率。 |
| 88 | <code>        g = len(self.weights)</code> | 终端外部接受因子广播为[g,size,2]并复制，以便反向递推修改。 |
| 89 | <code>        backward = np.broadcast_to(log_accept[None, :, None], (g, len(self.masks), 2)).copy()</code> | 终端外部接受因子广播为[g,size,2]并复制，以便反向递推修改。 |
| 90 | <code>        t00, t01, t10, t11 = (a[:, None] for a in self.transition)</code> | 转移按[g,1]广播到mask。 |
| 91 | <code>        result = np.empty(len(self.e))</code> | 为n个概率分配结果并逆序遍历。 |
| 92 | <code>        for t in range(len(self.e)-1, -1, -1):</code> | 为n个概率分配结果并逆序遍历。 |
| 93 | <code>            joint = self.forward[t]+backward+self.weights[:, None, None]</code> | 前向+后向+节点先验权重形成[g,size,2]联合量，仅最终边缘化时归一。 |
| 94 | <code>            zero, one = logsumexp(joint[:, :, 0]), logsumexp(joint[:, :, 1])</code> | 分别积掉节点/mask得到state0/state1的log质量。 |
| 95 | <code>            result[t] = np.exp(one-np.logaddexp(zero, one))</code> | one−logaddexp(zero,one)再exp，得到[0,1]状态概率，不返回log odds。 |
| 96 | <code>            if t:</code> | 还有上一位置时，state1路径需先把mask OR当前角色，再乘当前发射。 |
| 97 | <code>                next_one = backward[:, self.masks &#124; self.membership[t], 1]+self.e[t]</code> | 还有上一位置时，state1路径需先把mask OR当前角色，再乘当前发射。 |
| 98 | <code>                previous_zero = np.logaddexp(t00+backward[:, :, 0], t01+next_one)</code> | 分别按前状态0/1加00/01、10/11两种转移，state0路径不增mask。 |
| 99 | <code>                previous_one = np.logaddexp(t10+backward[:, :, 0], t11+next_one)</code> | 分别按前状态0/1加00/01、10/11两种转移，state0路径不增mask。 |
| 100 | <code>                backward = np.stack([previous_zero, previous_one], axis=-1)</code> | 沿末轴堆成[g,size,2]供上一时刻使用。 |
| 101 | <code>        if not np.isfinite(result).all():</code> | 拒绝非有限条件边缘，避免无质量接受事件传播NaN。 |
| 102 | <code>            raise ValueError('Nonfinite proof-conditioned marginal')</code> | 拒绝非有限条件边缘，避免无质量接受事件传播NaN。 |
| 103 | <code>        return result</code> | 返回片段内逐token state概率。 |
| 104 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 105 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 106 | <code>def _canonical_component(component, n):</code> | _canonical_component接收编译器一个不兼容DNF组件及有效token数，返回去冗余的角色token集合之合取项。 |
| 107 | <code>    roles = []</code> | 逐role处理其token索引列表。 |
| 108 | <code>    for tokens in component['role_tokens']:</code> | 逐role处理其token索引列表。 |
| 109 | <code>        if not tokens or any(not isinstance(t, (int, np.integer)) or not 0 &lt;= t &lt; n for t in tokens):</code> | role非空，所有索引须为整数且在有效评分轴[0,n)内；否则拒绝。 |
| 110 | <code>            raise ValueError('Role tokens must index the original scored axis')</code> | role非空，所有索引须为整数且在有效评分轴[0,n)内；否则拒绝。 |
| 111 | <code>        roles.append(tuple(sorted(set(map(int, tokens)))))</code> | 每role token去重排序为tuple，统一后续角色身份。 |
| 112 | <code>    terms = []</code> | 开始遍历repair_terms，每项是需要共同命中的role序号集合。 |
| 113 | <code>    for term in component['repair_terms']:</code> | 开始遍历repair_terms，每项是需要共同命中的role序号集合。 |
| 114 | <code>        if any(not isinstance(a, (int, np.integer)) or not 0 &lt;= a &lt; len(roles) for a in term):</code> | 修复项中的role索引必须有效。 |
| 115 | <code>            raise ValueError('Repair term contains an invalid role')</code> | 修复项中的role索引必须有效。 |
| 116 | <code>        terms.append(frozenset(roles[int(a)] for a in term))</code> | 把role序号映射为token tuple，再用frozenset表示一个不计顺序的合取项。 |
| 117 | <code>    if not terms:</code> | 不兼容组件必须存在可允许修复，空DNF拒绝。 |
| 118 | <code>        raise ValueError('An incompatible component must have admissible repairs')</code> | 不兼容组件必须存在可允许修复，空DNF拒绝。 |
| 119 | <code>    terms = set(terms)</code> | 合取项去重。 |
| 120 | <code>    return [term for term in terms if not any(other &lt; term for other in terms)]</code> | 删除严格超集项：单调DNF已有较小项时，较大项不改变事件接受集合。 |
| 121 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 122 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 123 | <code>def revise_scores(values, answer_index, unit_index, components, *, budget=np.log(2.),</code> | revise_scores 输入原d及回答/片段轴、DNF组件、KL与规模预算，返回bounded/hard/witness及诊断。proof相连片段按关联分量共享一个预算，不跨分量连块。 |
| 124 | <code>                  max_roles=8, max_terms=256):</code> | revise_scores 输入原d及回答/片段轴、DNF组件、KL与规模预算，返回bounded/hard/witness及诊断。proof相连片段按关联分量共享一个预算，不跨分量连块。 |
| 125 | <code>    """Join complete repair DNFs and revise each block-proof incidence component.</code> | revise_scores 输入原d及回答/片段轴、DNF组件、KL与规模预算，返回bounded/hard/witness及诊断。proof相连片段按关联分量共享一个预算，不跨分量连块。 |
| 126 | <code> </code> | revise_scores 输入原d及回答/片段轴、DNF组件、KL与规模预算，返回bounded/hard/witness及诊断。proof相连片段按关联分量共享一个预算，不跨分量连块。 |
| 127 | <code>    Unknown/budget-exceeded components are neutral, without dropping tokens.</code> | revise_scores 输入原d及回答/片段轴、DNF组件、KL与规模预算，返回bounded/hard/witness及诊断。proof相连片段按关联分量共享一个预算，不跨分量连块。 |
| 128 | <code>    Separate incidence components use separate budgets; they may share no block.</code> | revise_scores 输入原d及回答/片段轴、DNF组件、KL与规模预算，返回bounded/hard/witness及诊断。proof相连片段按关联分量共享一个预算，不跨分量连块。 |
| 129 | <code>    """</code> | revise_scores 输入原d及回答/片段轴、DNF组件、KL与规模预算，返回bounded/hard/witness及诊断。proof相连片段按关联分量共享一个预算，不跨分量连块。 |
| 130 | <code>    values, answers, units = map(np.asarray, (values, answer_index, unit_index))</code> | 三轴转数组并校验一维同形；不重排token。 |
| 131 | <code>    if values.ndim != 1 or answers.shape != values.shape or units.shape != values.shape:</code> | 三轴转数组并校验一维同形；不重排token。 |
| 132 | <code>        raise ValueError('Contrasts and original token axes must match')</code> | 三轴转数组并校验一维同形；不重排token。 |
| 133 | <code>    if not np.isfinite(values).all() or not np.isfinite(budget) or budget &lt; 0:</code> | 观测及KL预算必须有限，预算非负。 |
| 134 | <code>        raise ValueError('Finite contrasts and nonnegative budget required')</code> | 观测及KL预算必须有限，预算非负。 |
| 135 | <code>    if max_roles &lt; 1 or max_terms &lt; 1:</code> | role/term数量预算至少1，防止不存在可推断状态。 |
| 136 | <code>        raise ValueError('Positive inference budgets required')</code> | role/term数量预算至少1，防止不存在可推断状态。 |
| 137 | <code>    base = (grouped_persistent_scores(values, answers, units, witness=True)['score']</code> | 先计算原V3 witness odds；空输入显式跳过底层非空要求。 |
| 138 | <code>            if len(values) else np.empty(0))</code> | 先计算原V3 witness odds；空输入显式跳过底层非空要求。 |
| 139 | <code>    hard, bounded = expit(base), expit(base)</code> | hard和bounded分别从sigmoid(base)初始化，是不同数组，不把未覆盖token改为0。 |
| 140 | <code>    diagnostics = []</code> | 收集每个关联分量的应用/unknown原因。 |
| 141 | <code>    if not len(values) or not components:</code> | 没有token或组件就中立返回基础概率、基础odds及空诊断。 |
| 142 | <code>        return {'score': bounded, 'hard_score': hard, 'witness_score': base, 'components': diagnostics}</code> | 没有token或组件就中立返回基础概率、基础odds及空诊断。 |
| 143 | <code>    starts, stops = unit_ranges(answers, units)</code> | 取原连续块范围并将每token映射到块ID；这些块来自原分段。 |
| 144 | <code>    block_id = np.repeat(np.arange(len(starts)), stops-starts)</code> | 取原连续块范围并将每token映射到块ID；这些块来自原分段。 |
| 145 | <code>    prepared, seen = [], set()</code> | 标准化所有proof组件，同时准备去重集合。 |
| 146 | <code>    for component in components:</code> | 标准化所有proof组件，同时准备去重集合。 |
| 147 | <code>        terms = _canonical_component(component, len(values))</code> | 标准化所有proof组件，同时准备去重集合。 |
| 148 | <code>        if any(not term for term in terms):</code> | DNF含空合取时事件恒真，不应推动后验，故跳过。 |
| 149 | <code>            continue</code> | DNF含空合取时事件恒真，不应推动后验，故跳过。 |
| 150 | <code>        key = tuple(sorted(tuple(sorted(term)) for term in terms))</code> | 用规范排序key去重重复proof，避免重复约束重复用预算。 |
| 151 | <code>        if key in seen:</code> | 用规范排序key去重重复proof，避免重复约束重复用预算。 |
| 152 | <code>            continue</code> | 用规范排序key去重重复proof，避免重复约束重复用预算。 |
| 153 | <code>        seen.add(key)</code> | 用规范排序key去重重复proof，避免重复约束重复用预算。 |
| 154 | <code>        tokens = sorted({t for term in terms for role in term for t in role})</code> | 提取所有项涉及的有效token位置。 |
| 155 | <code>        if len(set(answers[tokens])) != 1:</code> | 一个proof不能跨回答，防止不同文本事实相互施加约束。 |
| 156 | <code>            raise ValueError('A proof cannot cross responses')</code> | 一个proof不能跨回答，防止不同文本事实相互施加约束。 |
| 157 | <code>        prepared.append((terms, set(block_id[tokens])))</code> | 保存DNF与其涉及的原始块集合。 |
| 158 | <code>    clusters = []</code> | 构造proof—block二部关系的连通分量。 |
| 159 | <code>    for index, (_, blocks) in enumerate(prepared):</code> | 构造proof—block二部关系的连通分量。 |
| 160 | <code>        hits = [i for i, (_, existing) in enumerate(clusters) if blocks &amp; existing]</code> | 当前proof若与已有分量共享任一块，逆序pop并合并所有相连分量，确保分量间没有共用块。 |
| 161 | <code>        indices, merged = [index], set(blocks)</code> | 当前proof若与已有分量共享任一块，逆序pop并合并所有相连分量，确保分量间没有共用块。 |
| 162 | <code>        for i in reversed(hits):</code> | 当前proof若与已有分量共享任一块，逆序pop并合并所有相连分量，确保分量间没有共用块。 |
| 163 | <code>            prior, touched = clusters.pop(i); indices.extend(prior); merged.update(touched)</code> | 当前proof若与已有分量共享任一块，逆序pop并合并所有相连分量，确保分量间没有共用块。 |
| 164 | <code>        clusters.append((indices, merged))</code> | 当前proof若与已有分量共享任一块，逆序pop并合并所有相连分量，确保分量间没有共用块。 |
| 165 | <code>    for indices, block_set in clusters:</code> | 逐连通分量收集proof因子与唯一token-role身份。 |
| 166 | <code>        terms_by_factor = [prepared[i][0] for i in indices]</code> | 逐连通分量收集proof因子与唯一token-role身份。 |
| 167 | <code>        roles = sorted({role for terms in terms_by_factor for term in terms for role in term})</code> | 逐连通分量收集proof因子与唯一token-role身份。 |
| 168 | <code>        metadata = {'blocks': sorted(map(int, block_set)), 'roles': len(roles),</code> | 记录块、role数、proof组件数和总term数，用于规模门槛/诊断。 |
| 169 | <code>                    'proof_components': len(indices), 'terms': sum(map(len, terms_by_factor))}</code> | 记录块、role数、proof组件数和总term数，用于规模门槛/诊断。 |
| 170 | <code>        if len(roles) &gt; max_roles or metadata['terms'] &gt; max_terms:</code> | 超角色或term预算时保持该分量原概率并记录unknown，不能删token或返回假的证明。 |
| 171 | <code>            diagnostics.append(metadata &#124; {'status': 'unknown', 'reason': 'combined_incidence_budget'})</code> | 超角色或term预算时保持该分量原概率并记录unknown，不能删token或返回假的证明。 |
| 172 | <code>            continue</code> | 超角色或term预算时保持该分量原概率并记录unknown，不能删token或返回假的证明。 |
| 173 | <code>        size = 1 &lt;&lt; len(roles)</code> | R个角色需要2^R个mask；为全回答token分配membership，未涉角色为0。 |
| 174 | <code>        masks, membership = np.arange(size), np.zeros(len(values), int)</code> | R个角色需要2^R个mask；为全回答token分配membership，未涉角色为0。 |
| 175 | <code>        for i, role in enumerate(roles):</code> | role对应的每token加一个bit；只要某token的latent state1出现就命中该role，并非role内所有token都为1。 |
| 176 | <code>            membership[list(role)] &#124;= 1 &lt;&lt; i</code> | role对应的每token加一个bit；只要某token的latent state1出现就命中该role，并非role内所有token都为1。 |
| 177 | <code>        role_bits = {role: 1 &lt;&lt; i for i, role in enumerate(roles)}</code> | 建立role tuple到单bit的映射。 |
| 178 | <code>        accepted = np.ones(size, bool)</code> | 所有mask初始接受，随后对每proof因子取AND。 |
| 179 | <code>        for terms in terms_by_factor:</code> | 每proof的term内角色AND、不同term OR、不同proof AND，产生完整联合DNF接受事件。 |
| 180 | <code>            term_masks = [sum(role_bits[role] for role in term) for term in terms]</code> | 每proof的term内角色AND、不同term OR、不同proof AND，产生完整联合DNF接受事件。 |
| 181 | <code>            accepted &amp;= np.any([(masks &amp; term) == term for term in term_masks], axis=0)</code> | 每proof的term内角色AND、不同term OR、不同proof AND，产生完整联合DNF接受事件。 |
| 182 | <code>        blocks = sorted(block_set)</code> | 为涉及的原块创建MaskedBlock；每块自己的π,c独立，跨块只通过事件连接。 |
| 183 | <code>        engines = [_MaskedBlock(values[starts[b]:stops[b]], membership[starts[b]:stops[b]], size)</code> | 为涉及的原块创建MaskedBlock；每块自己的π,c独立，跨块只通过事件连接。 |
| 184 | <code>                   for b in blocks]</code> | 为涉及的原块创建MaskedBlock；每块自己的π,c独立，跨块只通过事件连接。 |
| 185 | <code>        neutral = np.full(size, -np.inf); neutral[0] = 0.</code> | OR卷积单位元只在mask0有质量1，其余−Inf。 |
| 186 | <code>        prefix = [neutral]</code> | 左前缀卷积所有块的role命中分布。 |
| 187 | <code>        for engine in engines:</code> | 左前缀卷积所有块的role命中分布。 |
| 188 | <code>            prefix.append(_convolve(prefix[-1], engine.log_table))</code> | 左前缀卷积所有块的role命中分布。 |
| 189 | <code>        suffix = [neutral]</code> | 右后缀同样卷积并翻转，使每块能获得排除自身的其他块分布。 |
| 190 | <code>        for engine in reversed(engines):</code> | 右后缀同样卷积并翻转，使每块能获得排除自身的其他块分布。 |
| 191 | <code>            suffix.append(_convolve(engine.log_table, suffix[-1]))</code> | 右后缀同样卷积并翻转，使每块能获得排除自身的其他块分布。 |
| 192 | <code>        suffix.reverse()</code> | 右后缀同样卷积并翻转，使每块能获得排除自身的其他块分布。 |
| 193 | <code>        log_p = float(logsumexp(prefix[-1][accepted]))</code> | 在全块OR分布的accepted mask上求和，得到基础V3的事件log概率。 |
| 194 | <code>        if log_p &gt; 1e-10:</code> | 容忍微小舍入后要求事件质量≤1，并将微正log p裁0。 |
| 195 | <code>            raise ValueError('Event mass exceeds one')</code> | 容忍微小舍入后要求事件质量≤1，并将微正log p裁0。 |
| 196 | <code>        log_p = min(log_p, 0.)</code> | 容忍微小舍入后要求事件质量≤1，并将微正log p裁0。 |
| 197 | <code>        a, weight = bounded_event_probability(log_p, budget)</code> | 用固定KL预算求最大新事件概率a及hard条件混合权重，不把budget当proof可信度。 |
| 198 | <code>        for i, (b, engine) in enumerate(zip(blocks, engines, strict=True)):</code> | 逐块卷积左/右消息，得到排除该块的其他块mask分布。 |
| 199 | <code>            other = _convolve(prefix[i], suffix[i+1])</code> | 逐块卷积左/右消息，得到排除该块的其他块mask分布。 |
| 200 | <code>            accept_matrix = accepted[masks[:, None] &#124; masks[None, :]]</code> | 当前块mask与其他块mask做OR，查询是否满足完整事件。 |
| 201 | <code>            message = logsumexp(np.where(accept_matrix, other[None, :], -np.inf), axis=1)</code> | 对其他块接受质量边缘化，形成当前块每mask的外部log接受消息。 |
| 202 | <code>            hard[starts[b]:stops[b]] = engine.marginals(message)</code> | hard条件概率由块内精确状态边缘化计算，写回原token位置。 |
| 203 | <code>            bounded[starts[b]:stops[b]] = ((1-weight)*expit(base[starts[b]:stops[b]])</code> | bounded=(1−w)*基础概率+w*hard条件概率；没有改token身份或生成文字。 |
| 204 | <code>                                         +weight*hard[starts[b]:stops[b]])</code> | bounded=(1−w)*基础概率+w*hard条件概率；没有改token身份或生成文字。 |
| 205 | <code>        p = float(np.exp(log_p))</code> | 恢复基础事件p仅供诊断，罕见事件核心计算仍用log p。 |
| 206 | <code>        kl = float(xlogy(a, a)-a*log_p+xlogy(1-a, 1-a)</code> | 按Bernoulli分区计算实际D_KL(new||base)，p=1时为0。 |
| 207 | <code>                   -(1-a)*np.log1p(-p)) if p &lt; 1. else 0.</code> | 按Bernoulli分区计算实际D_KL(new||base)，p=1时为0。 |
| 208 | <code>        diagnostics.append(metadata &#124; {'status': 'applied', 'event_log_probability': log_p,</code> | 记录应用状态、基础/修订事件质量、混合权重、KL及a−p的概率改变量界。 |
| 209 | <code>            'revised_event_probability': a, 'hard_mixture_weight': weight, 'kl': kl,</code> | 记录应用状态、基础/修订事件质量、混合权重、KL及a−p的概率改变量界。 |
| 210 | <code>            'absolute_probability_shift_bound': a-p})</code> | 记录应用状态、基础/修订事件质量、混合权重、KL及a−p的概率改变量界。 |
| 211 | <code>    return {'score': bounded, 'hard_score': hard, 'witness_score': base, 'components': diagnostics}</code> | 返回有界诊断概率、hard条件概率、原witness odds及分量诊断；三条数值语义不同。 |
