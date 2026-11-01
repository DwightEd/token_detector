# myproject/graph.py 逐行讲解

基线：`1210120748b3a05bd271bc79debcbed4fdc0c964`；SHA-256：`4081fd8be3a8d662b5d6e02225ece2be3ad9f7ed4f2c9e8510aad1d418d6942d`；37 行。

源码与注释逐行并排。一个跨行表达式共用同一操作说明；这不代表各行独立执行。完整调用路径、输入输出轴和理论边界见 [METHOD.md](../METHOD.md)。未执行的 GPU 路径见 [验证边界](../VALIDATION.md)。

| 函数（含嵌套） | 原始行号 | 作用与契约入口 |
|---|---|---|
| `bounded_message` | 6–8 | bounded_message 输入邻居 log odds 和不一致惩罚，返回传过一条同状态偏好边的有限消息；由链的两个扫描方向共用。 |
| `chain_log_odds` | 11–23 | chain_log_odds 对能量 Σ emission*z−Σ penalty*[状态不同] 求逐位置后验 odds，时间和内存 O(N)，不是求 MAP 路径。 |
| `smooth_scores` | 26–37 | smooth_scores 输入 R4 连续分数、原 target 及回答轴，按 token 距离构链；默认 strength=.5，回答间断边。 |

| 行 | 原始代码 | 操作、目的与边界 |
|---|---|---|
| 1 | <code>"""Exact binary-chain messages on original answer positions."""</code> | 历史 R4 的二状态链精确消息，与共享 π,c 积分模型不同。 |
| 2 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 3 | <code>import numpy as np</code> | NumPy 提供 float64 消息向量与双曲函数。 |
| 4 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 5 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 6 | <code>def bounded_message(log_odds: float, penalty: float) -&gt; float:</code> | bounded_message 输入邻居 log odds 和不一致惩罚，返回传过一条同状态偏好边的有限消息；由链的两个扫描方向共用。 |
| 7 | <code>    """Return a same-state message bounded by the edge penalty."""</code> | bounded_message 输入邻居 log odds 和不一致惩罚，返回传过一条同状态偏好边的有限消息；由链的两个扫描方向共用。 |
| 8 | <code>    return float(2 * np.arctanh(np.tanh(penalty / 2) * np.tanh(log_odds / 2)))</code> | 解析 Ising 消息 2 atanh(tanh(J/2)tanh(h/2))，理论绝对值≤J（J≥0）；极端浮点 tanh 饱和仍可能溢出。 |
| 9 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 10 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 11 | <code>def chain_log_odds(emission: np.ndarray, penalties: np.ndarray) -&gt; np.ndarray:</code> | chain_log_odds 对能量 Σ emission*z−Σ penalty*[状态不同] 求逐位置后验 odds，时间和内存 O(N)，不是求 MAP 路径。 |
| 12 | <code>    """Solve sum(emission*z) - sum(penalty*[neighbor states differ])."""</code> | chain_log_odds 对能量 Σ emission*z−Σ penalty*[状态不同] 求逐位置后验 odds，时间和内存 O(N)，不是求 MAP 路径。 |
| 13 | <code>    emission = np.asarray(emission, dtype=np.float64)</code> | 统一发射与边惩罚为 float64。 |
| 14 | <code>    penalties = np.asarray(penalties, dtype=np.float64)</code> | 统一发射与边惩罚为 float64。 |
| 15 | <code>    if penalties.shape != (len(emission) - 1,):</code> | N 个位置必须恰有 N−1 条边；这里未单独验证一维、有限或非负惩罚。 |
| 16 | <code>        raise ValueError("One penalty is required between consecutive packed rows")</code> | N 个位置必须恰有 N−1 条边；这里未单独验证一维、有限或非负惩罚。 |
| 17 | <code>    left = np.zeros(len(emission), dtype=np.float64)</code> | 两端外部消息初始化为 0，对应链端无邻居影响。 |
| 18 | <code>    right = np.zeros_like(left)</code> | 两端外部消息初始化为 0，对应链端无邻居影响。 |
| 19 | <code>    for row in range(1, len(emission)):</code> | 左到右把上一位置 own emission 加已到达的左消息，再经当前边传播。 |
| 20 | <code>        left[row] = bounded_message(emission[row - 1] + left[row - 1], penalties[row - 1])</code> | 左到右把上一位置 own emission 加已到达的左消息，再经当前边传播。 |
| 21 | <code>    for row in range(len(emission) - 2, -1, -1):</code> | 右到左对称传播未来位置消息；所以此读出使用整条回答。 |
| 22 | <code>        right[row] = bounded_message(emission[row + 1] + right[row + 1], penalties[row])</code> | 右到左对称传播未来位置消息；所以此读出使用整条回答。 |
| 23 | <code>    return emission + left + right</code> | 每位置 own emission 加两个方向 cavity 消息，返回 [N] odds。 |
| 24 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 25 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 26 | <code>def smooth_scores(score: np.ndarray, target: np.ndarray,</code> | smooth_scores 输入 R4 连续分数、原 target 及回答轴，按 token 距离构链；默认 strength=.5，回答间断边。 |
| 27 | <code>                  answer_index: np.ndarray, strength: float = .5) -&gt; np.ndarray:</code> | smooth_scores 输入 R4 连续分数、原 target 及回答轴，按 token 距离构链；默认 strength=.5，回答间断边。 |
| 28 | <code>    """Apply R4 fixed continuity; gaps weaken edges and answers never connect."""</code> | smooth_scores 输入 R4 连续分数、原 target 及回答轴，按 token 距离构链；默认 strength=.5，回答间断边。 |
| 29 | <code>    if strength == 0:</code> | strength=0 直接复制原 score，保留历史消融行为；注意它不执行后面 2*(score−.5) 的尺度转换。 |
| 30 | <code>        return np.asarray(score, dtype=np.float64).copy()</code> | strength=0 直接复制原 score，保留历史消融行为；注意它不执行后面 2*(score−.5) 的尺度转换。 |
| 31 | <code>    same_answer = answer_index[1:] == answer_index[:-1]</code> | 只有相邻两行属于同一回答才允许连接。 |
| 32 | <code>    gaps = np.diff(np.asarray(target, dtype=np.int64))</code> | 距离用原 target 下标差，过滤掉 token 后不把它们当相邻原位置。 |
| 33 | <code>    if np.any(same_answer &amp; (gaps &lt;= 0)):</code> | 同回答 target 必须严格递增，否则拒绝，避免负距离或零除。 |
| 34 | <code>        raise ValueError("Original targets must increase within each answer")</code> | 同回答 target 必须严格递增，否则拒绝，避免负距离或零除。 |
| 35 | <code>    gate = np.zeros(len(target) - 1, dtype=np.float64)</code> | 边权初始为 0，跨回答边天然不传播。 |
| 36 | <code>    gate[same_answer] = 1 / gaps[same_answer]</code> | 同回答边用 1/gap，原位置缺口越大连续性越弱。 |
| 37 | <code>    return chain_log_odds(2.0 * (score - .5), strength * gate)</code> | 将 R4 分数线性映射为 unary log odds，再以 strength*gate 精确链推断；这里的线性映射是方法定义，不是事实概率校准。 |
