# myproject/sequence.py 逐行讲解

基线：`1210120748b3a05bd271bc79debcbed4fdc0c964`；SHA-256：`57c839c77eb93081bbde6d4f127497f62bc6586b41ac4b80fea8160d02953114`；218 行。

源码与注释逐行并排。一个跨行表达式共用同一操作说明；这不代表各行独立执行。完整调用路径、输入输出轴和理论边界见 [METHOD.md](../METHOD.md)。未执行的 GPU 路径见 [验证边界](../VALIDATION.md)。

| 函数（含嵌套） | 原始行号 | 作用与契约入口 |
|---|---|---|
| `_grid` | 16–43 | _grid(n,copying,prevalence) 返回 G 个 π,c、log 权重及四个转移概率的 log；供 persistent_scores 复用。缓存数组应被调用者视为只读。 |
| `_check_prevalence` | 46–50 | _check_prevalence 是 persistent/grouped 两入口的配置边界，成功无返回值。 |
| `witness_emissions` | 53–59 | witness_emissions 接收 d[N]，输出 log((q1+q0)/(2q1))；这是已有两个专家的嵌套混合，不增添事实观测。由 witness/onset/token-state 复用。 |
| `responsibility_logits` | 62–72 | responsibility_logits 输入原 d 和区域状态 odds，输出 absent 专家责任 odds；对应概率是 sigmoid(state)*sigmoid(d)，不是 factual probability。 |
| `persistent_scores` | 75–138 | persistent_scores 对一个非空片段推断，返回 token score、片段 evidence、persistence 诊断及 cavity 或 entry；由 grouped 与研究消融调用。 |
| `token_state_scores` | 141–161 | token_state_scores 用固定 P(z=1|h=0)=1/4、P(z=1|h=1)=3/4 分离持续 regime h 与 token z；解析积掉 z，再复用 persistent 推断。该固定通道不是从事实标签识别的。 |
| `adaptive_token_scores` | 164–184 | adaptive_token_scores 对共享和 token-deviation 两个结构给相同先验质量，用结构证据加权联合状态；等权后验版本仅是删除对照。 |
| `grouped_persistent_scores` | 187–217 | grouped_persistent_scores 接收拼接 d 与原回答/片段轴，返回同序分数及诊断；由 SharedDetector._score_values 调用。 |

| 行 | 原始代码 | 操作、目的与边界 |
|---|---|---|
| 1 | <code>"""Joint source-expert tendency and persistence, integrated without fitting.</code> | 模块声明：积分共享来源专家倾向和持续性；隐状态分配预测专家，保留原 token/片段，不能称为事实标签。 |
| 2 | <code> </code> | 模块声明：积分共享来源专家倾向和持续性；隐状态分配预测专家，保留原 token/片段，不能称为事实标签。 |
| 3 | <code>The latent states allocate predictive experts; they are not factuality labels.</code> | 模块声明：积分共享来源专家倾向和持续性；隐状态分配预测专家，保留原 token/片段，不能称为事实标签。 |
| 4 | <code>This readout retains the original token factor and span boundaries.</code> | 模块声明：积分共享来源专家倾向和持续性；隐状态分配预测专家，保留原 token/片段，不能称为事实标签。 |
| 5 | <code>"""</code> | 模块声明：积分共享来源专家倾向和持续性；隐状态分配预测专家，保留原 token/片段，不能称为事实标签。 |
| 6 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 7 | <code>from functools import lru_cache</code> | lru_cache 缓存只依赖 n/先验配置的求积网格，最多 128 项；不是回答结果缓存。 |
| 8 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 9 | <code>import numpy as np</code> | NumPy 管理节点、状态消息和输出轴。 |
| 10 | <code>from numpy.polynomial.legendre import leggauss</code> | leggauss 给出 [-1,1] 上正权重 Gauss–Legendre 求积节点。 |
| 11 | <code>from scipy.special import logsumexp</code> | logsumexp 稳定积分节点和隐状态的未归一化联合质量。 |
| 12 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 13 | <code>from .detector import unit_ranges</code> | 与其他读出共用原始连续回答/片段边界，不按状态后验重分段。 |
| 14 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 15 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 16 | <code>@lru_cache(maxsize=128)</code> | _grid(n,copying,prevalence) 返回 G 个 π,c、log 权重及四个转移概率的 log；供 persistent_scores 复用。缓存数组应被调用者视为只读。 |
| 17 | <code>def _grid(n, copying, prevalence='integrated'):</code> | _grid(n,copying,prevalence) 返回 G 个 π,c、log 权重及四个转移概率的 log；供 persistent_scores 复用。缓存数组应被调用者视为只读。 |
| 18 | <code>    # Polynomial degrees are &lt;=n in pi and &lt;=n-1 in c. Positive quadrature</code> | 证据关于 π 次数≤n、关于 c≤n−1；诊断 E[c] 多一次幂，matched 的 Beta(2,1) 权重再多一次，节点数包括这些预算。精确指代数积分，仍有浮点误差。 |
| 19 | <code>    # therefore integrates all evidence and marginal numerators exactly.</code> | 证据关于 π 次数≤n、关于 c≤n−1；诊断 E[c] 多一次幂，matched 的 Beta(2,1) 权重再多一次，节点数包括这些预算。精确指代数积分，仍有浮点误差。 |
| 20 | <code>    # The matched prior contributes 2*c, and the reported posterior c mean</code> | 证据关于 π 次数≤n、关于 c≤n−1；诊断 E[c] 多一次幂，matched 的 Beta(2,1) 权重再多一次，节点数包括这些预算。精确指代数积分，仍有浮点误差。 |
| 21 | <code>    # contributes one more power. Include that diagnostic in the exact budget.</code> | 证据关于 π 次数≤n、关于 c≤n−1；诊断 E[c] 多一次幂，matched 的 Beta(2,1) 权重再多一次，节点数包括这些预算。精确指代数积分，仍有浮点误差。 |
| 22 | <code>    count = 1 if prevalence == 'fixed' else (n+3)//2 if prevalence == 'matched' else n//2+1</code> | fixed 只需一个节点；matched 用 (n+3)//2；其他用 floor(n/2)+1，满足相应多项式精确次数。 |
| 23 | <code>    nodes, weights = leggauss(count)</code> | 产生 count 个节点及正权重，代数上积分最高 2*count−1 次多项式。 |
| 24 | <code>    pi, weights = (nodes+1)/2, weights/2</code> | 仿射移到 [0,1]，权重除 2 包含区间 Jacobian。 |
| 25 | <code>    if prevalence == 'integrated':</code> | 默认同时对 uniform π,c 积分；copying=False 时 c 退化为 0，恢复 Beta(1,1) 交换模型。 |
| 26 | <code>        c, c_weights = (pi, weights) if copying else (np.array([0.]), np.array([1.]))</code> | 默认同时对 uniform π,c 积分；copying=False 时 c 退化为 0，恢复 Beta(1,1) 交换模型。 |
| 27 | <code>    elif prevalence in {'symmetric', 'matched'} and copying:</code> | symmetric/matched 保留 c 网格而固定 π 的后续设置。 |
| 28 | <code>        c, c_weights = pi, weights</code> | symmetric/matched 保留 c 网格而固定 π 的后续设置。 |
| 29 | <code>        if prevalence == 'matched':</code> | matched 给 c 权重乘 Beta(2,1) 密度 2c，不重新等权平均。 |
| 30 | <code>            c_weights = c_weights*(2*c)</code> | matched 给 c 权重乘 Beta(2,1) 密度 2c，不重新等权平均。 |
| 31 | <code>        pi, weights = np.array([.5]), np.array([1.])</code> | 两种模式都令 π=.5，移除 prevalence 积分维度。 |
| 32 | <code>    elif prevalence == 'fixed' and copying:</code> | fixed 为标准固定 HMM：π 已由单节点得到 .5，c 固定 2/3。 |
| 33 | <code>        c, c_weights = np.array([2/3]), np.array([1.])</code> | fixed 为标准固定 HMM：π 已由单节点得到 .5，c 固定 2/3。 |
| 34 | <code>    elif prevalence == 'fixed_persistence' and copying:</code> | fixed_persistence 保留 uniform π 网格，c 固定 .5。 |
| 35 | <code>        c, c_weights = np.array([.5]), np.array([1.])</code> | fixed_persistence 保留 uniform π 网格，c 固定 .5。 |
| 36 | <code>    else:</code> | 未知配置或未声明的非复制先验组合报错，不自动替代先验。 |
| 37 | <code>        raise ValueError('Unknown prevalence prior or fixed prevalence without copying')</code> | 未知配置或未声明的非复制先验组合报错，不自动替代先验。 |
| 38 | <code>    pi, c = np.meshgrid(pi, c, indexing='ij')</code> | π,c 做笛卡尔网格，[Qπ,Qc]，索引 ij 保持外积权重顺序。 |
| 39 | <code>    pi, c = pi.ravel(), c.ravel()</code> | 展平成长度 G=Qπ*Qc 的节点轴。 |
| 40 | <code>    log_weights = np.log(np.multiply.outer(weights, c_weights).ravel())</code> | 先验节点权重外积后取 log，与展平后的 π,c 一一对应。 |
| 41 | <code>    transition = np.array([(1-pi)+c*pi, (1-c)*pi,</code> | 四行依次为 T00=1−π+cπ、T01=(1−c)π、T10=(1−c)(1−π)、T11=c+(1−c)π，每行长度 G。 |
| 42 | <code>                           (1-c)*(1-pi), c+(1-c)*pi])</code> | 四行依次为 T00=1−π+cπ、T01=(1−c)π、T10=(1−c)(1−π)、T11=c+(1−c)π，每行长度 G。 |
| 43 | <code>    return pi, c, log_weights, np.log(transition)</code> | 返回节点、log 先验权重和 log 转移；转移 log 不改变概率归一性。 |
| 44 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 45 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 46 | <code>def _check_prevalence(prevalence, *, witness, copying=True, onset=False):</code> | _check_prevalence 是 persistent/grouped 两入口的配置边界，成功无返回值。 |
| 47 | <code>    if prevalence not in {'integrated', 'symmetric', 'matched', 'fixed', 'fixed_persistence'}:</code> | 拒绝不存在的 prevalence 模型名。 |
| 48 | <code>        raise ValueError('Unknown prevalence prior')</code> | 拒绝不存在的 prevalence 模型名。 |
| 49 | <code>    if prevalence != 'integrated' and (not witness or not copying or onset):</code> | 非默认先验只定义了 witness+copying 且无 onset 的组合，禁止混用产生未定义模型。 |
| 50 | <code>        raise ValueError('Nondefault prior requires witness emissions and copying, without onset')</code> | 非默认先验只定义了 witness+copying 且无 onset 的组合，禁止混用产生未定义模型。 |
| 51 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 52 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 53 | <code>def witness_emissions(values):</code> | witness_emissions 接收 d[N]，输出 log((q1+q0)/(2q1))；这是已有两个专家的嵌套混合，不增添事实观测。由 witness/onset/token-state 复用。 |
| 54 | <code>    """Likelihood ratio of (p_source+p_absent)/2 relative to p_source.</code> | witness_emissions 接收 d[N]，输出 log((q1+q0)/(2q1))；这是已有两个专家的嵌套混合，不增添事实观测。由 witness/onset/token-state 复用。 |
| 55 | <code> </code> | witness_emissions 接收 d[N]，输出 log((q1+q0)/(2q1))；这是已有两个专家的嵌套混合，不增添事实观测。由 witness/onset/token-state 复用。 |
| 56 | <code>    A risky region can still contain source-predictable tokens. This nested</code> | witness_emissions 接收 d[N]，输出 log((q1+q0)/(2q1))；这是已有两个专家的嵌套混合，不增添事实观测。由 witness/onset/token-state 复用。 |
| 57 | <code>    alternative adds no information and does not identify factual errors.</code> | witness_emissions 接收 d[N]，输出 log((q1+q0)/(2q1))；这是已有两个专家的嵌套混合，不增添事实观测。由 witness/onset/token-state 复用。 |
| 58 | <code>    """</code> | witness_emissions 接收 d[N]，输出 log((q1+q0)/(2q1))；这是已有两个专家的嵌套混合，不增添事实观测。由 witness/onset/token-state 复用。 |
| 59 | <code>    return np.logaddexp(0., np.asarray(values, float))-np.log(2.)</code> | 稳定计算 log(1+exp d)−log2，下界 −log2，d=0 时为 0。 |
| 60 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 61 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 62 | <code>def responsibility_logits(values, state_scores):</code> | responsibility_logits 输入原 d 和区域状态 odds，输出 absent 专家责任 odds；对应概率是 sigmoid(state)*sigmoid(d)，不是 factual probability。 |
| 63 | <code>    """Posterior absent-expert log odds inside the declared witness hierarchy.</code> | responsibility_logits 输入原 d 和区域状态 odds，输出 absent 专家责任 odds；对应概率是 sigmoid(state)*sigmoid(d)，不是 factual probability。 |
| 64 | <code> </code> | responsibility_logits 输入原 d 和区域状态 odds，输出 absent 专家责任 odds；对应概率是 sigmoid(state)*sigmoid(d)，不是 factual probability。 |
| 65 | <code>    P(absent&#124;d)=sigmoid(state_scores)*sigmoid(d). This is a predictive</code> | responsibility_logits 输入原 d 和区域状态 odds，输出 absent 专家责任 odds；对应概率是 sigmoid(state)*sigmoid(d)，不是 factual probability。 |
| 66 | <code>    allocation diagnostic, not a probability of factual error.</code> | responsibility_logits 输入原 d 和区域状态 odds，输出 absent 专家责任 odds；对应概率是 sigmoid(state)*sigmoid(d)，不是 factual probability。 |
| 67 | <code>    """</code> | responsibility_logits 输入原 d 和区域状态 odds，输出 absent 专家责任 odds；对应概率是 sigmoid(state)*sigmoid(d)，不是 factual probability。 |
| 68 | <code>    values, state_scores = np.asarray(values, float), np.asarray(state_scores, float)</code> | 两个输入统一为浮点向量。 |
| 69 | <code>    if (values.ndim != 1 or state_scores.shape != values.shape</code> | 要求一维、同形、有限，禁止不同 token 轴广播掩盖错位。 |
| 70 | <code>            or not np.isfinite(values).all() or not np.isfinite(state_scores).all()):</code> | 要求一维、同形、有限，禁止不同 token 轴广播掩盖错位。 |
| 71 | <code>        raise ValueError('Matching finite contrast and state-score vectors required')</code> | 要求一维、同形、有限，禁止不同 token 轴广播掩盖错位。 |
| 72 | <code>    return values+state_scores-np.logaddexp(0., np.logaddexp(values, state_scores))</code> | 稳定 logit 乘积概率：d+s−log(1+exp d+exp s)，避免先算接近 0/1 的概率再除。 |
| 73 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 74 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 75 | <code>def persistent_scores(values, *, copying=True, witness=False, onset=False,</code> | persistent_scores 对一个非空片段推断，返回 token score、片段 evidence、persistence 诊断及 cavity 或 entry；由 grouped 与研究消融调用。 |
| 76 | <code>                      prevalence='integrated'):</code> | persistent_scores 对一个非空片段推断，返回 token score、片段 evidence、persistence 诊断及 cavity 或 entry；由 grouped 与研究消融调用。 |
| 77 | <code>    """Exact two-state forward/backward marginalization under explicit priors.</code> | persistent_scores 对一个非空片段推断，返回 token score、片段 evidence、persistence 诊断及 cavity 或 entry；由 grouped 与研究消融调用。 |
| 78 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 79 | <code>    P(z_t=1&#124;z_previous,pi,c)=c*z_previous+(1-c)*pi.</code> | 给定 π,c 时是二状态马尔可夫链；c=0 恢复交换共享。必须先积分联合证据，再归一化，不能平均各网格后验。 |
| 80 | <code>    copying=False fixes c=0, recovering exchangeable Beta(1,1) sharing.</code> | 给定 π,c 时是二状态马尔可夫链；c=0 恢复交换共享。必须先积分联合证据，再归一化，不能平均各网格后验。 |
| 81 | <code>    Integrate unnormalized joints, not uniformly averaged node posteriors.</code> | 给定 π,c 时是二状态马尔可夫链；c=0 恢复交换共享。必须先积分联合证据，再归一化，不能平均各网格后验。 |
| 82 | <code>    onset=True uses the absent expert at 0-&gt;1 entries (including initial 1),</code> | onset 在 0→1 和初始 1 用原 absent 比，在 1→1 用半源混合；不读取 gold 起点。 |
| 83 | <code>    and the half-source mixture at 1-&gt;1 continuation. No gold onset is used.</code> | onset 在 0→1 和初始 1 用原 absent 比，在 1→1 用半源混合；不读取 gold 起点。 |
| 84 | <code>    prevalence='symmetric' fixes pi=.5 with uniform c; 'matched' fixes pi=.5</code> | 文档列出 symmetric、matched、fixed、fixed_persistence 的不同先验，并说明它们匹配的相邻同状态先验概率 5/6；匹配这一矩不代表模型相同。 |
| 85 | <code>    with c~Beta(2,1), matching the integrated model's adjacent agreement 5/6.</code> | 文档列出 symmetric、matched、fixed、fixed_persistence 的不同先验，并说明它们匹配的相邻同状态先验概率 5/6；匹配这一矩不代表模型相同。 |
| 86 | <code>    prevalence='fixed' fixes pi=.5 and c=2/3: a standard fixed HMM with</code> | 文档列出 symmetric、matched、fixed、fixed_persistence 的不同先验，并说明它们匹配的相邻同状态先验概率 5/6；匹配这一矩不代表模型相同。 |
| 87 | <code>    the same adjacent agreement 5/6. prevalence='fixed_persistence' retains</code> | 文档列出 symmetric、matched、fixed、fixed_persistence 的不同先验，并说明它们匹配的相邻同状态先验概率 5/6；匹配这一矩不代表模型相同。 |
| 88 | <code>    uniform pi and fixes c=.5, also preserving adjacent agreement 5/6.</code> | 文档列出 symmetric、matched、fixed、fixed_persistence 的不同先验，并说明它们匹配的相邻同状态先验概率 5/6；匹配这一矩不代表模型相同。 |
| 89 | <code>    All nondefault prevalence modes require</code> | 非默认先验受 witness/copying 约束；任何模式都没有自动得到事实监督。 |
| 90 | <code>    witness emissions and copying; none introduces a factuality observation.</code> | 非默认先验受 witness/copying 约束；任何模式都没有自动得到事实监督。 |
| 91 | <code>    """</code> | 非默认先验受 witness/copying 约束；任何模式都没有自动得到事实监督。 |
| 92 | <code>    _check_prevalence(prevalence, witness=witness, copying=copying, onset=onset)</code> | 先验证先验和发射组合。 |
| 93 | <code>    values = np.asarray(values, float)</code> | 将片段 d 转 float（NumPy 默认 float64）。 |
| 94 | <code>    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():</code> | 拒绝非一维、空和非有限 d，确保 log 消息有定义。 |
| 95 | <code>        raise ValueError('Nonempty finite token contrasts required')</code> | 拒绝非一维、空和非有限 d，确保 log 消息有定义。 |
| 96 | <code>    if onset and witness:</code> | onset 与 witness 互斥，防止同时替换入口与全部状态发射。 |
| 97 | <code>        raise ValueError('Choose witness or onset emissions, not both')</code> | onset 与 witness 互斥，防止同时替换入口与全部状态发射。 |
| 98 | <code>    continuation = witness_emissions(values) if onset else values</code> | onset 的持续状态单独保存 witness emission；其他情况下 continuation 就是原 d。 |
| 99 | <code>    if witness:</code> | witness 模式将全部 state=1 发射替换为混合比 w(d)，state=0 因子仍为 1。 |
| 100 | <code>        values = witness_emissions(values)</code> | witness 模式将全部 state=1 发射替换为混合比 w(d)，state=0 因子仍为 1。 |
| 101 | <code>    n = len(values)</code> | n 是片段 token 数。 |
| 102 | <code>    pi, c, weights, transition = _grid(n, copying, prevalence)</code> | 取 G 个 π,c 节点、log 权重与转移；默认 G=O(n²)。 |
| 103 | <code>    t00, t01, t10, t11 = transition</code> | 按 00/01/10/11 取四个长度 G 的 log 转移。 |
| 104 | <code>    forward = np.empty((n, len(pi), 2))</code> | 分配 [n,G,2] 前向 log 联合表，主导内存 O(nG)。 |
| 105 | <code>    forward[0, :, 0], forward[0, :, 1] = np.log1p(-pi), np.log(pi)+values[0]</code> | 初始状态概率是 [1−π,π]；只有 state=1 加首 token 发射，保存每节点未归一化质量。 |
| 106 | <code>    for t in range(1, n):</code> | 按 token 向前扫描，previous 是 [G,2] 的上一时刻联合质量。 |
| 107 | <code>        previous = forward[t-1]</code> | 按 token 向前扫描，previous 是 [G,2] 的上一时刻联合质量。 |
| 108 | <code>        forward[t, :, 0] = np.logaddexp(previous[:, 0]+t00, previous[:, 1]+t10)</code> | 到 state=0 的两条路径 0→0、1→0 作 logaddexp，state0 无额外 d。 |
| 109 | <code>        forward[t, :, 1] = (np.logaddexp(previous[:, 0]+t01+values[t],</code> | onset 时到 state1 的两条路径分别加原 d（入口）与 continuation（持续），不能提出共同发射。 |
| 110 | <code>                                        previous[:, 1]+t11+continuation[t]) if onset else</code> | onset 时到 state1 的两条路径分别加原 d（入口）与 continuation（持续），不能提出共同发射。 |
| 111 | <code>                            values[t]+np.logaddexp(previous[:, 0]+t01, previous[:, 1]+t11))</code> | 普通模式到 state1 的两条转移先相加，再统一乘 exp(values[t])。 |
| 112 | <code>    node_evidence = np.logaddexp(forward[-1, :, 0], forward[-1, :, 1])</code> | 每个节点对最终 0/1 状态求和，得到该节点完整片段 evidence。 |
| 113 | <code>    evidence = float(logsumexp(weights+node_evidence))</code> | 加 log 求积权重后再 logsumexp，得到对 π,c 积分后的 evidence。 |
| 114 | <code>    persistence = .5 if prevalence == 'fixed_persistence' else float(np.exp(weights+node_evidence-evidence) @ c)</code> | 按 evidence 更新节点权重再计算 E[c]；fixed_persistence 直接返回其固定 .5。 |
| 115 | <code>    backward, score = np.zeros((len(pi), 2)), np.empty(n)</code> | 后向终端为 [G,2] 的零 log 因子，另分配 n 个 token odds。 |
| 116 | <code>    entry_score = np.empty(n) if onset else None</code> | 只有 onset 才分配 n 个入口事件 odds。 |
| 117 | <code>    for t in range(n-1, -1, -1):</code> | 从 n−1 到 0 反向扫描，有限结束。 |
| 118 | <code>        joints = forward[t]+backward+weights[:, None]</code> | 前向、后向、节点先验权重三者相加，形成当前时刻 [G,2] 联合质量。 |
| 119 | <code>        score[t] = logsumexp(joints[:, 1])-logsumexp(joints[:, 0])</code> | 分别对两状态积分，再相减得到 token log posterior odds；未显式计算概率也完成归一化比。 |
| 120 | <code>        if onset:</code> | onset 的 t=0 入口事件等同初始 state1，entry_score=score。 |
| 121 | <code>            if t == 0:</code> | onset 的 t=0 入口事件等同初始 state1，entry_score=score。 |
| 122 | <code>                entry_score[t] = score[t]</code> | onset 的 t=0 入口事件等同初始 state1，entry_score=score。 |
| 123 | <code>            else:</code> | 其他 t 的入口仅为 0→1，组合上一时刻前向、T01、原 d_t 与后向 state1。 |
| 124 | <code>                previous = forward[t-1]</code> | 其他 t 的入口仅为 0→1，组合上一时刻前向、T01、原 d_t 与后向 state1。 |
| 125 | <code>                entry = previous[:, 0]+t01+values[t]+backward[:, 1]</code> | 其他 t 的入口仅为 0→1，组合上一时刻前向、T01、原 d_t 与后向 state1。 |
| 126 | <code>                other = np.stack((previous[:, 0]+t00+backward[:, 0],</code> | 非入口包括 0→0、1→0 和 1→1，持续路径使用 continuation，而非原 d。 |
| 127 | <code>                                  previous[:, 1]+t10+backward[:, 0],</code> | 非入口包括 0→0、1→0 和 1→1，持续路径使用 continuation，而非原 d。 |
| 128 | <code>                                  previous[:, 1]+t11+continuation[t]+backward[:, 1]))</code> | 非入口包括 0→0、1→0 和 1→1，持续路径使用 continuation，而非原 d。 |
| 129 | <code>                entry_score[t] = logsumexp(weights+entry)-logsumexp(weights+other)</code> | 入口联合质量对全部非入口联合质量取 log 比，是事件 odds，不是 state1 边缘 odds。 |
| 130 | <code>        next_zero = np.logaddexp(t00+backward[:, 0], t01+values[t]+backward[:, 1])</code> | 更新上一时刻 state0 的后向消息：走到当前0或当前1，后者用入口原 d。 |
| 131 | <code>        next_one = np.logaddexp(t10+backward[:, 0],</code> | 更新上一时刻 state1：走到当前0，或继续1并使用相应持续发射。 |
| 132 | <code>                               t11+(continuation[t] if onset else values[t])+backward[:, 1])</code> | 更新上一时刻 state1：走到当前0，或继续1并使用相应持续发射。 |
| 133 | <code>        backward = np.column_stack([next_zero, next_one])</code> | 将两种前状态消息堆叠为 [G,2]，供下一个倒序位置使用。 |
| 134 | <code>    if not np.isfinite(score).all():</code> | 最终 score 若非有限则拒绝，不让 NaN 进入后续评价。 |
| 135 | <code>        raise ValueError('Nonfinite persistent-state responsibility')</code> | 最终 score 若非有限则拒绝，不让 NaN 进入后续评价。 |
| 136 | <code>    result = {'score': score, 'log_evidence': evidence, 'persistence_mean': persistence}</code> | 收集逐 token odds 和两个片段级诊断。 |
| 137 | <code>    result.update({'entry_score': entry_score} if onset else {'cavity_log_odds': score-values})</code> | onset 返回 entry；其他模式 cavity=score−实际使用的发射，witness 时减的是 w(d) 而非原 d。 |
| 138 | <code>    return result</code> | 返回本片段结果；不会保存到跨回答对象状态。 |
| 139 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 140 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 141 | <code>def token_state_scores(values):</code> | token_state_scores 用固定 P(z=1|h=0)=1/4、P(z=1|h=1)=3/4 分离持续 regime h 与 token z；解析积掉 z，再复用 persistent 推断。该固定通道不是从事实标签识别的。 |
| 142 | <code>    """Separate a persistent regime from individual states with a fixed channel.</code> | token_state_scores 用固定 P(z=1|h=0)=1/4、P(z=1|h=1)=3/4 分离持续 regime h 与 token z；解析积掉 z，再复用 persistent 推断。该固定通道不是从事实标签识别的。 |
| 143 | <code> </code> | token_state_scores 用固定 P(z=1|h=0)=1/4、P(z=1|h=1)=3/4 分离持续 regime h 与 token z；解析积掉 z，再复用 persistent 推断。该固定通道不是从事实标签识别的。 |
| 144 | <code>    P(z=1&#124;h=0)=1/4 and P(z=1&#124;h=1)=3/4. Integrate z analytically, infer h</code> | token_state_scores 用固定 P(z=1|h=0)=1/4、P(z=1|h=1)=3/4 分离持续 regime h 与 token z；解析积掉 z，再复用 persistent 推断。该固定通道不是从事实标签识别的。 |
| 145 | <code>    with the existing kernel, then recover z from the leave-one-out message.</code> | token_state_scores 用固定 P(z=1|h=0)=1/4、P(z=1|h=1)=3/4 分离持续 regime h 与 token z；解析积掉 z，再复用 persistent 推断。该固定通道不是从事实标签识别的。 |
| 146 | <code>    Context log odds are bounded by +/-log(3); own witness evidence is retained.</code> | token_state_scores 用固定 P(z=1|h=0)=1/4、P(z=1|h=1)=3/4 分离持续 regime h 与 token z；解析积掉 z，再复用 persistent 推断。该固定通道不是从事实标签识别的。 |
| 147 | <code>    Neither state is a calibrated factuality label.</code> | token_state_scores 用固定 P(z=1|h=0)=1/4、P(z=1|h=1)=3/4 分离持续 regime h 与 token z；解析积掉 z，再复用 persistent 推断。该固定通道不是从事实标签识别的。 |
| 148 | <code>    """</code> | token_state_scores 用固定 P(z=1|h=0)=1/4、P(z=1|h=1)=3/4 分离持续 regime h 与 token z；解析积掉 z，再复用 persistent 推断。该固定通道不是从事实标签识别的。 |
| 149 | <code>    values = np.asarray(values, float)</code> | 统一浮点并验证非空有限一维观测。 |
| 150 | <code>    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():</code> | 统一浮点并验证非空有限一维观测。 |
| 151 | <code>        raise ValueError('Nonempty finite token contrasts required')</code> | 统一浮点并验证非空有限一维观测。 |
| 152 | <code>    emission = witness_emissions(values)</code> | 个体 token 的 self evidence 仍使用 witness 混合比。 |
| 153 | <code>    low, high = np.log(.25), np.log(.75)</code> | 以 log .25 和 log .75 表示固定状态通道概率。 |
| 154 | <code>    zero = np.logaddexp(high, low+emission)</code> | regime0 对 z 边缘化的 log 发射 log(.75+.25*exp(w))，用于恢复 evidence。 |
| 155 | <code>    contrast = np.logaddexp(low-emission, high)-np.logaddexp(high-emission, low)</code> | 计算 regime1 相对 regime0 的 log 发射比，用减 w 的形式避免巨大 exp；值被通道限制。 |
| 156 | <code>    result = persistent_scores(contrast)</code> | 对 regime 发射调用同一个 persistent 内核，不复制 HMM 实现。 |
| 157 | <code>    cavity = result['cavity_log_odds']</code> | 从 regime 后验取去掉自身发射的 cavity。 |
| 158 | <code>    context = np.logaddexp(low, high+cavity)-np.logaddexp(high, low+cavity)</code> | 通过固定噪声通道把 regime cavity 映射为 token context odds，限制在 ±log3。 |
| 159 | <code>    return {'score': emission+context, 'cavity_log_odds': context,</code> | 最终 token odds=own witness emission+有界 context，同时暴露 context。 |
| 160 | <code>            'log_evidence': result['log_evidence']+float(zero.sum()),</code> | 补回前面被比值化除掉的 regime0 发射总和，才能比较模型 evidence。 |
| 161 | <code>            'persistence_mean': result['persistence_mean']}</code> | 返回 regime persistence 诊断，而不是个体 z 的事实持续概率。 |
| 162 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 163 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 164 | <code>def adaptive_token_scores(values):</code> | adaptive_token_scores 对共享和 token-deviation 两个结构给相同先验质量，用结构证据加权联合状态；等权后验版本仅是删除对照。 |
| 165 | <code>    """Integrate equal prior mass over shared and token-deviation structures.</code> | adaptive_token_scores 对共享和 token-deviation 两个结构给相同先验质量，用结构证据加权联合状态；等权后验版本仅是删除对照。 |
| 166 | <code> </code> | adaptive_token_scores 对共享和 token-deviation 两个结构给相同先验质量，用结构证据加权联合状态；等权后验版本仅是删除对照。 |
| 167 | <code>    Combine unnormalized state joints using each structure's evidence. The</code> | adaptive_token_scores 对共享和 token-deviation 两个结构给相同先验质量，用结构证据加权联合状态；等权后验版本仅是删除对照。 |
| 168 | <code>    uniform-posterior mixture is returned only as a frozen deletion control.</code> | adaptive_token_scores 对共享和 token-deviation 两个结构给相同先验质量，用结构证据加权联合状态；等权后验版本仅是删除对照。 |
| 169 | <code>    Structure evidence is predictive compatibility, not factual correctness.</code> | adaptive_token_scores 对共享和 token-deviation 两个结构给相同先验质量，用结构证据加权联合状态；等权后验版本仅是删除对照。 |
| 170 | <code>    """</code> | adaptive_token_scores 对共享和 token-deviation 两个结构给相同先验质量，用结构证据加权联合状态；等权后验版本仅是删除对照。 |
| 171 | <code>    shared = persistent_scores(values, witness=True)</code> | 计算 witness 共享结构的状态 odds/evidence。 |
| 172 | <code>    token = token_state_scores(values)</code> | 计算固定通道 token-state 结构的 odds/evidence。 |
| 173 | <code>    scores = np.stack([shared['score'], token['score']])</code> | 将两结构 token odds 堆成 [2,n]。 |
| 174 | <code>    evidence = np.array([shared['log_evidence'], token['log_evidence']])</code> | 结构 evidence 是 [2]，之后应对其做后验加权。 |
| 175 | <code>    zero, one = -np.logaddexp(0., scores), -np.logaddexp(0., -scores)</code> | 稳定计算两结构下 token state0/state1 的 log 概率，仍为 [2,n]。 |
| 176 | <code>    score = logsumexp(evidence[:, None]+one, axis=0)-logsumexp(evidence[:, None]+zero, axis=0)</code> | 将每结构归一化概率乘其 evidence，再对结构求和，分别得到 state1/0 联合质量的 log 比。 |
| 177 | <code>    probability = float(np.exp(evidence[1]-logsumexp(evidence)))</code> | 计算 token-deviation 结构的片段级后验概率，不能解释为发生事实错误的概率。 |
| 178 | <code>    return {'score': score, 'witness_score': shared['score'], 'token_state_score': token['score'],</code> | 返回组合 score，并保存两结构原始 token 分数。 |
| 179 | <code>            'uniform_model_score': logsumexp(one, axis=0)-logsumexp(zero, axis=0),</code> | uniform_model_score 故意不加 evidence，是供消融检验的等权后验混合。 |
| 180 | <code>            'deviation_probability': probability,</code> | deviation_probability 是每片段一个结构概率，grouped 包装会广播到每 token。 |
| 181 | <code>            'cavity_log_odds': score-witness_emissions(values),</code> | 输出组合 odds 去掉公共 own witness emission 的差值诊断。 |
| 182 | <code>            'log_evidence': float(logsumexp(evidence)-np.log(2.)),</code> | 结构 evidence 求和后减 log2，包含等先验模型质量。 |
| 183 | <code>            'persistence_mean': (1-probability)*shared['persistence_mean']</code> | 用结构后验权重平均两种 persistence_mean，得到混合诊断。 |
| 184 | <code>                                +probability*token['persistence_mean']}</code> | 用结构后验权重平均两种 persistence_mean，得到混合诊断。 |
| 185 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 186 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 187 | <code>def grouped_persistent_scores(values, answer_index, unit_index, *, witness=False, onset=False,</code> | grouped_persistent_scores 接收拼接 d 与原回答/片段轴，返回同序分数及诊断；由 SharedDetector._score_values 调用。 |
| 188 | <code>                              token_state=False, adaptive=False, prevalence='integrated'):</code> | grouped_persistent_scores 接收拼接 d 与原回答/片段轴，返回同序分数及诊断；由 SharedDetector._score_values 调用。 |
| 189 | <code>    """Keep original row order and reset the latent state at each original span."""</code> | grouped_persistent_scores 接收拼接 d 与原回答/片段轴，返回同序分数及诊断；由 SharedDetector._score_values 调用。 |
| 190 | <code>    _check_prevalence(prevalence, witness=witness, onset=onset)</code> | 先校验先验/发射设置，copying 在此包装固定为 True。 |
| 191 | <code>    if prevalence != 'integrated' and (token_state or adaptive):</code> | 非默认 prevalence 不能用于 token_state/adaptive，防止组合没有声明的先验。 |
| 192 | <code>        raise ValueError('Fixed prevalence does not define token-state or adaptive compositions')</code> | 非默认 prevalence 不能用于 token_state/adaptive，防止组合没有声明的先验。 |
| 193 | <code>    if token_state and (witness or onset):</code> | token_state 自己定义 witness 通道，禁止再叠 witness/onset。 |
| 194 | <code>        raise ValueError('Token-state inference defines its own witness emissions')</code> | token_state 自己定义 witness 通道，禁止再叠 witness/onset。 |
| 195 | <code>    if adaptive and (witness or onset or token_state):</code> | adaptive 已内部选择两个结构，禁止同时要求其他结构标志。 |
| 196 | <code>        raise ValueError('Adaptive inference defines its own structures')</code> | adaptive 已内部选择两个结构，禁止同时要求其他结构标志。 |
| 197 | <code>    values, answer_index, unit_index = map(np.asarray, (values, answer_index, unit_index))</code> | 三个输入转数组，不排序或合并身份。 |
| 198 | <code>    if values.ndim != 1 or answer_index.shape != values.shape or unit_index.shape != values.shape:</code> | 验证一维与同长；有限性由每片段实际内核检查。 |
| 199 | <code>        raise ValueError('Original token axes must match contrasts')</code> | 验证一维与同长；有限性由每片段实际内核检查。 |
| 200 | <code>    starts, stops = unit_ranges(answer_index, unit_index)</code> | 按连续 answer/unit 变化切范围；此低层不单独支持空片段，SharedDetector 先处理空值。 |
| 201 | <code>    score, persistence = np.empty(len(values)), []</code> | 预分配 n 个 score，片段 persistence 用列表收集。 |
| 202 | <code>    entry = np.empty(len(values)) if onset else None</code> | onset 的 entry 是 token 轴向量。 |
| 203 | <code>    extras = {key: np.empty(len(values)) for key in ['witness_score', 'token_state_score',</code> | adaptive 才分配四条额外 token 轴诊断；其中片段结构概率后续会广播。 |
| 204 | <code>              'uniform_model_score', 'deviation_probability']} if adaptive else {}</code> | adaptive 才分配四条额外 token 轴诊断；其中片段结构概率后续会广播。 |
| 205 | <code>    for a, b in zip(starts, stops, strict=True):</code> | 严格按成对边界遍历，每片段重置隐状态。 |
| 206 | <code>        result = (adaptive_token_scores(values[a:b]) if adaptive else</code> | adaptive 优先选择结构混合内核，前面互斥检查确保无歧义。 |
| 207 | <code>                  token_state_scores(values[a:b]) if token_state else</code> | 否则 token_state 选择固定通道内核。 |
| 208 | <code>                  persistent_scores(values[a:b], witness=witness, onset=onset,</code> | 其余情况调用 persistent 并传 witness/onset/prevalence。 |
| 209 | <code>                                    prevalence=prevalence))</code> | 其余情况调用 persistent 并传 witness/onset/prevalence。 |
| 210 | <code>        score[a:b] = result['score']</code> | 分数写回原 token 范围。 |
| 211 | <code>        persistence.append(result['persistence_mean'])</code> | persistence 每片段收集一次，长度不同于 token 数。 |
| 212 | <code>        if onset:</code> | onset 额外写回逐 token 入口分数。 |
| 213 | <code>            entry[a:b] = result['entry_score']</code> | onset 额外写回逐 token 入口分数。 |
| 214 | <code>        for key, output in extras.items():</code> | adaptive 诊断逐字段写回；标量结构概率依 NumPy 广播填整片段。 |
| 215 | <code>            output[a:b] = result[key]</code> | adaptive 诊断逐字段写回；标量结构概率依 NumPy 广播填整片段。 |
| 216 | <code>    return {'score': score, 'persistence_mean': np.asarray(persistence),</code> | 返回 token score、片段 persistence 和对应模式字段，不把所有模式强行填同一假 schema。 |
| 217 | <code>            **extras, **({'entry_score': entry} if onset else {})}</code> | 返回 token score、片段 persistence 和对应模式字段，不把所有模式强行填同一假 schema。 |
| 218 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
