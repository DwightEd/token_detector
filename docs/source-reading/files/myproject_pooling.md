# myproject/pooling.py 逐行讲解

基线：`1210120748b3a05bd271bc79debcbed4fdc0c964`；SHA-256：`80e852b64e301cf6007893df3b61d058ef39a97c81c561416f2b2917c63bef49`；153 行。

源码与注释逐行并排。一个跨行表达式共用同一操作说明；这不代表各行独立执行。完整调用路径、输入输出轴和理论边界见 [METHOD.md](../METHOD.md)。未执行的 GPU 路径见 [验证边界](../VALIDATION.md)。

| 函数（含嵌套） | 原始行号 | 作用与契约入口 |
|---|---|---|
| `_values` | 8–12 | _values 是 forward、cavity 和轴检查的输入边界：接收类数组，返回有限非空的一维 float64 对比。 |
| `_parameters` | 15–17 | _parameters 验证两个 Beta 形状参数；成功仅返回 None，无状态更改。 |
| `_axes` | 20–25 | _axes 供 grouped_scores/population_spans 使用，联合校验对比与任意多个 token 元数据轴。 |
| `forward` | 28–38 | forward(values) 输出 [n+1,n+1] log 系数；第 t 行第 k 列是前 t 个位置恰有 k 个 absent 状态的质量。供 cavity 与 fit_prior 共用。 |
| `_terminal` | 41–43 | _terminal(n,a,b) 将每个总计数映射到对 π 积分后的 log 权重。 |
| `cavity_scores` | 46–66 | cavity_scores 接收一个片段的 d 和 a,b，返回逐 token log odds、排除自身的修正、一个片段证据；默认 a=b=1。由 grouped_scores 调用。 |
| `prior_objective` | 69–90 | prior_objective 供 BFGS 调用：输入 log(a,b)、各片段末行系数和权重，输出负 MAP 目标与 log 坐标梯度。 |
| `fit_prior` | 93–116 | fit_prior 从未标注片段估计 a,b 两个群体超参数，返回参数和每次优化诊断；并非训练神经网络。 |
| `fit_prior.<lambda>@112:29` | 112–112 | 在合格候选中选 MAP 目标最大的一个；不是按测试标签选参。 |
| `grouped_scores` | 119–131 | grouped_scores 接收拼接 token 与回答/片段轴，逐连续片段调用 cavity，返回同顺序分数和每片段证据。SharedDetector 默认调用这里。 |
| `population_spans` | 134–153 | population_spans 给经验 Bayes 构造片段及来源均衡权重；输入四条 token 轴，返回片段列表和权重向量，不改变原 token。 |

| 行 | 原始代码 | 操作、目的与边界 |
|---|---|---|
| 1 | <code>"""Exact Beta-mixture pooling of unchanged per-token source log contrasts."""</code> | 模块定位：对未改变的逐 token 来源 log 对比作精确 Beta 混合推断，不是对事实标签训练的分类器。 |
| 2 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 3 | <code>import numpy as np</code> | NumPy 承担 float64 向量、三角 DP 表和轴索引。 |
| 4 | <code>from scipy.optimize import minimize</code> | SciPy minimize 用于可选的两个先验参数 MAP 优化；默认 shared 推断不会执行优化。 |
| 5 | <code>from scipy.special import betaln, digamma, logsumexp</code> | betaln 稳定计算 Beta 积分；digamma 计算其导数；logsumexp 稳定边缘化状态。 |
| 6 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 7 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 8 | <code>def _values(values):</code> | _values 是 forward、cavity 和轴检查的输入边界：接收类数组，返回有限非空的一维 float64 对比。 |
| 9 | <code>    values = np.asarray(values, dtype=np.float64)</code> | 转 float64，统一 DP 和积分精度；可能复用已有同 dtype 数组，不承诺独立副本。 |
| 10 | <code>    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():</code> | 拒绝二维、空或 NaN/Inf 输入；这些输入不能形成此处的有限片段后验。 |
| 11 | <code>        raise ValueError('Expected a nonempty finite token contrast vector')</code> | 拒绝二维、空或 NaN/Inf 输入；这些输入不能形成此处的有限片段后验。 |
| 12 | <code>    return values</code> | 将验证后的 n 维对比向量交还调用者。 |
| 13 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 14 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 15 | <code>def _parameters(alpha, beta):</code> | _parameters 验证两个 Beta 形状参数；成功仅返回 None，无状态更改。 |
| 16 | <code>    if not np.isfinite([alpha, beta]).all() or min(alpha, beta) &lt;= 0:</code> | Beta 密度要求 a,b 正且有限；越界立即 ValueError，避免 betaln 后续产生无意义结果。 |
| 17 | <code>        raise ValueError('Beta parameters must be positive and finite')</code> | Beta 密度要求 a,b 正且有限；越界立即 ValueError，避免 betaln 后续产生无意义结果。 |
| 18 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 19 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 20 | <code>def _axes(contrasts, *indices):</code> | _axes 供 grouped_scores/population_spans 使用，联合校验对比与任意多个 token 元数据轴。 |
| 21 | <code>    contrasts = _values(contrasts)</code> | 复用对比验证；因此此包装层也拒绝空输入。 |
| 22 | <code>    axes = [np.asarray(index) for index in indices]</code> | 将每个身份/分段轴转成 NumPy 数组，不强制整数类型或身份唯一性。 |
| 23 | <code>    if any(axis.ndim != 1 or len(axis) != len(contrasts) for axis in axes):</code> | 逐轴要求一维且长度 n；这里只检查形状，身份一致性在 population_spans 额外检查。 |
| 24 | <code>        raise ValueError('Token metadata must be one-dimensional and match contrast length')</code> | 逐轴要求一维且长度 n；这里只检查形状，身份一致性在 population_spans 额外检查。 |
| 25 | <code>    return contrasts, *axes</code> | 返回一个元组，顺序是对比后接传入的各轴，保持原始 token 行序。 |
| 26 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 27 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 28 | <code>def forward(values):</code> | forward(values) 输出 [n+1,n+1] log 系数；第 t 行第 k 列是前 t 个位置恰有 k 个 absent 状态的质量。供 cavity 与 fit_prior 共用。 |
| 29 | <code>    """Log Bernstein coefficients: sum over assignments with k absent-world tokens."""</code> | forward(values) 输出 [n+1,n+1] log 系数；第 t 行第 k 列是前 t 个位置恰有 k 个 absent 状态的质量。供 cavity 与 fit_prior 共用。 |
| 30 | <code>    values = _values(values)</code> | 先确保输入是有限、非空的一维 float64。 |
| 31 | <code>    n = len(values)</code> | n 是当前片段长度，不是整个数据集的 token 数。 |
| 32 | <code>    result = np.full((n + 1, n + 1), -np.inf)</code> | 分配平方表并用 -Inf 表示不可达计数；只填下三角有效区，内存 O(n²)。 |
| 33 | <code>    result[0, 0] = 0.</code> | 空前缀、零 absent 状态的权重为 1，其 log 为 0。 |
| 34 | <code>    for t, value in enumerate(values):</code> | 逐位置推进，t 从 0 到 n−1，value 是当前 d_t。 |
| 35 | <code>        result[t+1, :t+2] = np.logaddexp(</code> | 对两种状态分支作 logaddexp，并把新前缀写到第 t+1 行。 |
| 36 | <code>            np.r_[result[t, :t+1], -np.inf],</code> | 当前状态为 0 时保持 absent 计数 k 不变；右端补 -Inf 处理不可达的最大计数。 |
| 37 | <code>            np.r_[-np.inf, result[t, :t+1] + value])</code> | 当前状态为 1 时旧计数右移一格并加 d_t；左端 -Inf 禁止负计数。 |
| 38 | <code>    return result</code> | 返回所有前缀系数以便 cavity 重用；fit_prior 只取末行。 |
| 39 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 40 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 41 | <code>def _terminal(n, alpha, beta):</code> | _terminal(n,a,b) 将每个总计数映射到对 π 积分后的 log 权重。 |
| 42 | <code>    count = np.arange(n + 1)</code> | 枚举 K=0,…,n 的所有可能 absent 个数。 |
| 43 | <code>    return betaln(alpha + count, beta + n - count) - betaln(alpha, beta)</code> | 计算 log B(a+K,b+n−K)−log B(a,b)，已经包含归一化 Beta 先验，不再加二项系数。 |
| 44 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 45 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 46 | <code>def cavity_scores(values, alpha=1., beta=1.):</code> | cavity_scores 接收一个片段的 d 和 a,b，返回逐 token log odds、排除自身的修正、一个片段证据；默认 a=b=1。由 grouped_scores 调用。 |
| 47 | <code>    """Exact token log-responsibility odds; not a calibrated factual-error posterior.</code> | 文档明确分数是专家责任 odds；其他位置只通过 cavity 贡献，同一个已生成回答历史被固定，不能称为事实错误后验。 |
| 48 | <code> </code> | 文档明确分数是专家责任 odds；其他位置只通过 cavity 贡献，同一个已生成回答历史被固定，不能称为事实错误后验。 |
| 49 | <code>    The own contrast is never averaged. Other observed factors provide its</code> | 文档明确分数是专家责任 odds；其他位置只通过 cavity 贡献，同一个已生成回答历史被固定，不能称为事实错误后验。 |
| 50 | <code>    cavity prior; autoregressive histories remain the original fixed histories.</code> | 文档明确分数是专家责任 odds；其他位置只通过 cavity 贡献，同一个已生成回答历史被固定，不能称为事实错误后验。 |
| 51 | <code>    """</code> | 文档明确分数是专家责任 odds；其他位置只通过 cavity 贡献，同一个已生成回答历史被固定，不能称为事实错误后验。 |
| 52 | <code>    _parameters(alpha, beta)</code> | 先检查先验合法，防止不正参数进入积分。 |
| 53 | <code>    values = _values(values)</code> | 转并校验当前片段观测 d。 |
| 54 | <code>    n = len(values)</code> | 记录当前片段长度以分配消息表。 |
| 55 | <code>    prefix = forward(values)</code> | 构建 prefix[t,k]，只包含当前位置之前的观测，尚未乘 Beta 终端。 |
| 56 | <code>    suffix = np.full_like(prefix, -np.inf)</code> | 为反向消息分配同形状 float64 表，未定义状态初始化 -Inf。 |
| 57 | <code>    suffix[n] = _terminal(n, alpha, beta)</code> | 末行填所有总计数的 Beta 积分，给反向递推提供终止条件。 |
| 58 | <code>    for t in range(n-1, -1, -1):</code> | 从 n−1 倒数到 0，循环有限次，不涉及迭代收敛。 |
| 59 | <code>        suffix[t, :t+1] = np.logaddexp(suffix[t+1, :t+1],</code> | z_t=0 时剩余消息保持累计 k 不变。 |
| 60 | <code>                                                  values[t] + suffix[t+1, 1:t+2])</code> | z_t=1 时加本位置 d 并把累计 k 增一，与上一行的 0 分支稳定相加。 |
| 61 | <code>    correction = np.empty(n)</code> | 为 n 个 cavity log odds 分配输出向量。 |
| 62 | <code>    for t in range(n):</code> | 逐 token 计算排除该 token 自证据的状态比。 |
| 63 | <code>        correction[t] = (logsumexp(prefix[t, :t+1] + suffix[t+1, 1:t+2])</code> | 分子：该 token 固定为 1，拼接前缀计数与后缀 k+1；没有加本位置 d_t。 |
| 64 | <code>                         - logsumexp(prefix[t, :t+1] + suffix[t+1, :t+1]))</code> | 分母：该 token 固定为 0，拼接相同前缀与后缀 k；两项 logsumexp 之差即 cavity odds。 |
| 65 | <code>    return {'score': values + correction, 'cavity_log_odds': correction,</code> | 最终 score=d+correction；不把 token 证据替换为片段平均，同时返回 correction 以便审查。 |
| 66 | <code>            'log_evidence': float(suffix[0, 0])}</code> | suffix[0,0] 已边缘化所有状态与 π，是当前片段相对基准因子的 log evidence。 |
| 67 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 68 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 69 | <code>def prior_objective(log_parameters, coefficients, weights):</code> | prior_objective 供 BFGS 调用：输入 log(a,b)、各片段末行系数和权重，输出负 MAP 目标与 log 坐标梯度。 |
| 70 | <code>    """Source-weighted composite evidence plus fixed independent Gamma(2,1) priors.</code> | 目标是按来源加权的复合证据加两个 Gamma(2,1) 先验；密度相对于 a,b，换优化坐标不改变 MAP 目标，因此不额外加 Jacobian。 |
| 71 | <code> </code> | 目标是按来源加权的复合证据加两个 Gamma(2,1) 先验；密度相对于 a,b，换优化坐标不改变 MAP 目标，因此不额外加 Jacobian。 |
| 72 | <code>    The MAP density is with respect to alpha,beta. Log parameters are only an</code> | 目标是按来源加权的复合证据加两个 Gamma(2,1) 先验；密度相对于 a,b，换优化坐标不改变 MAP 目标，因此不额外加 Jacobian。 |
| 73 | <code>    optimization coordinate; no transformation Jacobian is added to this target.</code> | 目标是按来源加权的复合证据加两个 Gamma(2,1) 先验；密度相对于 a,b，换优化坐标不改变 MAP 目标，因此不额外加 Jacobian。 |
| 74 | <code>    """</code> | 目标是按来源加权的复合证据加两个 Gamma(2,1) 先验；密度相对于 a,b，换优化坐标不改变 MAP 目标，因此不额外加 Jacobian。 |
| 75 | <code>    alpha, beta = np.exp(log_parameters)</code> | 指数映射保证优化变量对应正 a,b；过大导致溢出会由后续验证拒绝。 |
| 76 | <code>    _parameters(alpha, beta)</code> | 验证指数映射后的先验参数仍有限且为正。 |
| 77 | <code>    objective = float(np.log(alpha) - alpha + np.log(beta) - beta)</code> | 两个 Gamma(2,1) log 密度忽略常数后为 log a−a+log b−b。 |
| 78 | <code>    gradient = np.array([1/alpha - 1, 1/beta - 1], dtype=float)</code> | 初始化 a,b 原坐标梯度 [1/a−1,1/b−1]。 |
| 79 | <code>    for coefficient, weight in zip(coefficients, weights, strict=True):</code> | 严格配对片段系数和权重；数量不匹配不能悄悄截断。 |
| 80 | <code>        n = len(coefficient) - 1</code> | 末行系数有 n+1 项，由此恢复片段长度。 |
| 81 | <code>        count = np.arange(n + 1)</code> | 为该片段构造所有 K 计数。 |
| 82 | <code>        terms = coefficient + _terminal(n, alpha, beta)</code> | 把观测系数和 Beta 终端相加，得到各 K 的 log 联合质量。 |
| 83 | <code>        evidence = logsumexp(terms)</code> | 对 K 边缘化得到片段 log evidence。 |
| 84 | <code>        posterior = np.exp(terms - evidence)</code> | 归一化各 K 的后验权重，用来取 digamma 导数期望。 |
| 85 | <code>        objective += weight * evidence</code> | 将当前片段证据按外部来源权重加入目标，权重不是标签。 |
| 86 | <code>        common = -digamma(alpha + beta + n) + digamma(alpha + beta)</code> | a,b 导数共同包含总浓度 digamma 的差。 |
| 87 | <code>        gradient += weight * np.array([</code> | 把该片段加权导数累加到两个先验参数梯度。 |
| 88 | <code>            posterior @ digamma(alpha + count) - digamma(alpha) + common,</code> | a 导数使用后验 E[ψ(a+K)]−ψ(a) 加共同浓度项。 |
| 89 | <code>            posterior @ digamma(beta + n - count) - digamma(beta) + common])</code> | b 导数对称地使用 E[ψ(b+n−K)]−ψ(b)。 |
| 90 | <code>    return -objective, -gradient * np.array([alpha, beta])</code> | SciPy 做最小化所以取负；乘 [a,b] 是从原参数到 log 参数的链式求导，不是密度 Jacobian。 |
| 91 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 92 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 93 | <code>def fit_prior(spans, weights):</code> | fit_prior 从未标注片段估计 a,b 两个群体超参数，返回参数和每次优化诊断；并非训练神经网络。 |
| 94 | <code>    """Fit only two population hyperparameters; no labels or neural optimization."""</code> | fit_prior 从未标注片段估计 a,b 两个群体超参数，返回参数和每次优化诊断；并非训练神经网络。 |
| 95 | <code>    coefficients = [forward(values)[-1] for values in spans]</code> | 预先计算每片段的末行系数，BFGS 每一步复用，不重复 token DP。 |
| 96 | <code>    weights = np.asarray(weights, dtype=float)</code> | 权重转浮点数组，用于复合似然。 |
| 97 | <code>    if (not coefficients or weights.shape != (len(coefficients),)</code> | 要求至少一个片段、一片段一个严格正有限权重；空群体或错轴立即拒绝。 |
| 98 | <code>            or not np.isfinite(weights).all() or np.any(weights &lt;= 0)):</code> | 要求至少一个片段、一片段一个严格正有限权重；空群体或错轴立即拒绝。 |
| 99 | <code>        raise ValueError('Each nonempty span needs a positive finite population weight')</code> | 要求至少一个片段、一片段一个严格正有限权重；空群体或错轴立即拒绝。 |
| 100 | <code>    attempts = []</code> | 收集所有优化起点的实际结果，包含失败记录。 |
| 101 | <code>    for start in [(1., 1.), (.5, 2.), (2., .5)]:</code> | 固定三个起点避免单一起点依赖；仍没有全局最优保证。 |
| 102 | <code>        result = minimize(prior_objective, np.log(start), args=(coefficients, weights),</code> | 在 log 坐标执行带解析梯度的 BFGS，最多 500 步，内部 gtol=1e−8。 |
| 103 | <code>                          method='BFGS', jac=True, options={'gtol': 1e-8, 'maxiter': 500})</code> | 在 log 坐标执行带解析梯度的 BFGS，最多 500 步，内部 gtol=1e−8。 |
| 104 | <code>        value, gradient = prior_objective(result.x, coefficients, weights)</code> | 独立重算终点目标与梯度，不只相信优化器 success 标志。 |
| 105 | <code>        attempts.append({'start': list(start), 'parameters': np.exp(result.x).tolist(),</code> | 记录起点和恢复到原坐标的 a,b。 |
| 106 | <code>                         'objective': float(-value), 'gradient_max': float(np.max(abs(gradient))),</code> | 记录原 MAP 目标与 log 坐标最大绝对梯度，供冻结验收门槛使用。 |
| 107 | <code>                         'success': bool(result.success), 'iterations': int(result.nit),</code> | 保留优化器 success 与实际迭代次数，便于区分数值退出原因。 |
| 108 | <code>                         'message': str(result.message)})</code> | 保存优化器原始消息，不把失败静默变成成功。 |
| 109 | <code>    accepted = [a for a in attempts if np.isfinite(a['objective']) and a['gradient_max'] &lt;= 1e-6]</code> | 只有有限目标且终点梯度 ≤1e−6 的候选被接受；success 字段本身不是决定条件。 |
| 110 | <code>    if not accepted:</code> | 三个起点都不合格时抛 RuntimeError，并附诊断，不退回未经声明的默认值。 |
| 111 | <code>        raise RuntimeError(f'No prior fit met the frozen gradient tolerance: {attempts}')</code> | 三个起点都不合格时抛 RuntimeError，并附诊断，不退回未经声明的默认值。 |
| 112 | <code>    best = max(accepted, key=lambda a: a['objective'])</code> | 在合格候选中选 MAP 目标最大的一个；不是按测试标签选参。 |
| 113 | <code>    alpha, beta = best['parameters']</code> | 取最佳候选的两个形状参数。 |
| 114 | <code>    return {'alpha': alpha, 'beta': beta, 'mean': alpha/(alpha+beta),</code> | 输出 a,b 和先验均值 a/(a+b)。 |
| 115 | <code>            'concentration': alpha+beta, 'objective': best['objective'],</code> | 浓度 a+b 控制共享程度，并返回最佳目标。 |
| 116 | <code>            'population_weight': float(weights.sum()), 'attempts': attempts}</code> | 附总群体权重及全部尝试记录，便于复查实际拟合。 |
| 117 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 118 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 119 | <code>def grouped_scores(contrasts, answer_index, unit_index, alpha=1., beta=1.):</code> | grouped_scores 接收拼接 token 与回答/片段轴，逐连续片段调用 cavity，返回同顺序分数和每片段证据。SharedDetector 默认调用这里。 |
| 120 | <code>    """Keep original row order; states never cross an answer or span boundary."""</code> | grouped_scores 接收拼接 token 与回答/片段轴，逐连续片段调用 cavity，返回同顺序分数和每片段证据。SharedDetector 默认调用这里。 |
| 121 | <code>    from myproject.detector import unit_ranges</code> | 延迟引入连续边界函数，复用 R4 的轴分段实现。 |
| 122 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 123 | <code>    contrasts, answer_index, unit_index = _axes(contrasts, answer_index, unit_index)</code> | 同时校验三条一维同长度轴，不排序输入。 |
| 124 | <code>    starts, stops = unit_ranges(answer_index, unit_index)</code> | 只在回答身份或 unit 身份发生变化的位置切段；身份相同但不相邻也不会连接。 |
| 125 | <code>    score, correction, evidence = np.empty_like(contrasts), np.empty_like(contrasts), []</code> | 预分配 n 个分数与修正，并用列表收集每片段证据。 |
| 126 | <code>    for start, stop in zip(starts, stops):</code> | 依照连续半开区间遍历片段。 |
| 127 | <code>        result = cavity_scores(contrasts[start:stop], alpha, beta)</code> | 每片段独立精确推断，共享状态不跨片段或回答边界。 |
| 128 | <code>        score[start:stop] = result['score']</code> | 将片段最终 log odds 写回原 token 行位置。 |
| 129 | <code>        correction[start:stop] = result['cavity_log_odds']</code> | 将对应 cavity 修正写回原位置。 |
| 130 | <code>        evidence.append(result['log_evidence'])</code> | 片段证据每段一项，不能误认成逐 token 轴。 |
| 131 | <code>    return {'score': score, 'cavity_log_odds': correction, 'log_evidence': np.asarray(evidence)}</code> | 返回两个长度 n 的数组与长度片段数的证据数组。 |
| 132 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 133 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 134 | <code>def population_spans(contrasts, answer_index, unit_index, source_index):</code> | population_spans 给经验 Bayes 构造片段及来源均衡权重；输入四条 token 轴，返回片段列表和权重向量，不改变原 token。 |
| 135 | <code>    """One unit of fit weight/source, shared over its answers and their spans."""</code> | population_spans 给经验 Bayes 构造片段及来源均衡权重；输入四条 token 轴，返回片段列表和权重向量，不改变原 token。 |
| 136 | <code>    from myproject.detector import unit_ranges</code> | 同样复用连续边界逻辑。 |
| 137 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 138 | <code>    contrasts, answer_index, unit_index, source_index = _axes(</code> | 先验证 contrasts、answer、unit、source 四条轴长度一致。 |
| 139 | <code>        contrasts, answer_index, unit_index, source_index)</code> | 先验证 contrasts、answer、unit、source 四条轴长度一致。 |
| 140 | <code>    for answer in np.unique(answer_index):</code> | 逐全局 answer 身份检查来源归属。 |
| 141 | <code>        if len(np.unique(source_index[answer_index == answer])) != 1:</code> | 一个 answer 身份若同时属于多个 source 则拒绝，否则来源均衡权重会混淆独立单位。 |
| 142 | <code>            raise ValueError('Each global answer identity must belong to one source')</code> | 一个 answer 身份若同时属于多个 source 则拒绝，否则来源均衡权重会混淆独立单位。 |
| 143 | <code>    starts, stops = unit_ranges(answer_index, unit_index)</code> | 找出各连续回答/片段范围。 |
| 144 | <code>    spans, weights = [], []</code> | 初始化返回片段与权重容器。 |
| 145 | <code>    for start, stop in zip(starts, stops):</code> | 按原顺序访问每个片段。 |
| 146 | <code>        source, answer = source_index[start], answer_index[start]</code> | 用片段起点取该段来源与回答身份。 |
| 147 | <code>        if not np.all(source_index[start:stop] == source):</code> | 段内来源不一致时拒绝，禁止跨来源共享。 |
| 148 | <code>            raise ValueError('A span cannot cross a source')</code> | 段内来源不一致时拒绝，禁止跨来源共享。 |
| 149 | <code>        answers = np.unique(answer_index[source_index == source])</code> | 统计该来源的唯一回答数，防止多回答来源获得更大总权重。 |
| 150 | <code>        span_count = np.sum(answer_index[starts] == answer)</code> | 统计当前回答的片段数，用来在其片段之间均分回答权重。 |
| 151 | <code>        spans.append(contrasts[start:stop])</code> | 保存当前片段对比的切片，仍然不含事实标签。 |
| 152 | <code>        weights.append(1 / (len(answers) * span_count))</code> | 每段权重=1/(该来源回答数×该回答片段数)，故每来源总权重恰为 1。 |
| 153 | <code>    return spans, np.asarray(weights)</code> | 输出供 fit_prior 使用的片段与 float 权重数组。 |
