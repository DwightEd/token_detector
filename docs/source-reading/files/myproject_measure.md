# myproject/measure.py 逐行讲解

基线：`1210120748b3a05bd271bc79debcbed4fdc0c964`；SHA-256：`0778b233b07a42c1d1a5e6739027394ab5c0644b25cca3082e8b53998eedbbd3`；71 行。

源码与注释逐行并排。一个跨行表达式共用同一操作说明；这不代表各行独立执行。完整调用路径、输入输出轴和理论边界见 [METHOD.md](../METHOD.md)。未执行的 GPU 路径见 [验证边界](../VALIDATION.md)。

| 函数（含嵌套） | 原始行号 | 作用与契约入口 |
|---|---|---|
| `token_scores` | 8–15 | token_scores 输入 measurement pack、原始编码和等长分数，返回 JSON 兼容列表；由 API.predict 和实验输出调用。 |
| `window_mean` | 18–26 | window_mean 是历史离线 route 的长度 16 均值：包含自身、前 7、后 8；它使用未来位置，不能称实时检测。 |
| `canonical_delta` | 29–41 | canonical_delta 重演历史的单元均值加偏差运算，数学上等于原值，但浮点舍入不必逐位相同。 |
| `answer_axis` | 44–60 | answer_axis 从编码建立有效 token、回答、片段身份；独立于测量了哪些信号，是所有读出共用的坐标契约。 |
| `answer_rows` | 63–71 | answer_rows 将原始观测过滤到有效轴并附元数据；供 API.measure 和离线测量调用。 |

| 行 | 原始代码 | 操作、目的与边界 |
|---|---|---|
| 1 | <code>"""In-memory measurement alignment on the original response token positions."""</code> | 只负责内存中的观测与原始回答轴对齐，不加载模型或读取标签。 |
| 2 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 3 | <code>import numpy as np</code> | NumPy 用于有效 token mask、平滑与字段拼接。 |
| 4 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 5 | <code>from .distribution import DISTRIBUTION_FIELDS</code> | 可选完整词表诊断字段集合与 distribution 模块保持一致。 |
| 6 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 7 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 8 | <code>def token_scores(pack: dict, response: dict, scores: np.ndarray) -&gt; list[dict]:</code> | token_scores 输入 measurement pack、原始编码和等长分数，返回 JSON 兼容列表；由 API.predict 和实验输出调用。 |
| 9 | <code>    """Render scores on the existing original token/character axis, without I/O."""</code> | token_scores 输入 measurement pack、原始编码和等长分数，返回 JSON 兼容列表；由 API.predict 和实验输出调用。 |
| 10 | <code>    result = []</code> | 为每个有效 token 准备一条记录。 |
| 11 | <code>    for target, token_id, score in zip(pack['target'], pack['token_id'], scores, strict=True):</code> | target、token_id、score 严格同时迭代，长度不等即报错，不悄悄截断。 |
| 12 | <code>        start, end = map(int, response['offsets'][target])</code> | target 索引原始回答 offset；即使过滤了中间 token 也不重排字符坐标。 |
| 13 | <code>        result.append({'token_id': int(token_id), 'text': response['text'][start:end],</code> | 切原文并将 NumPy 标量转 Python int/float；text 来自原文字符区间而不是 decode。 |
| 14 | <code>                       'start': start, 'end': end, 'score': float(score)})</code> | 切原文并将 NumPy 标量转 Python int/float；text 来自原文字符区间而不是 decode。 |
| 15 | <code>    return result</code> | 返回最终有效 token 列表，不写文件。 |
| 16 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 17 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 18 | <code>def window_mean(values: np.ndarray, window: int = 16) -&gt; np.ndarray:</code> | window_mean 是历史离线 route 的长度 16 均值：包含自身、前 7、后 8；它使用未来位置，不能称实时检测。 |
| 19 | <code>    """Historical offline route window: seven past and eight future positions."""</code> | window_mean 是历史离线 route 的长度 16 均值：包含自身、前 7、后 8；它使用未来位置，不能称实时检测。 |
| 20 | <code>    values = np.asarray(values, dtype=np.float64)</code> | 统一到 float64 以复现历史路线统计。 |
| 21 | <code>    target = np.arange(len(values))</code> | 在原观测轴上构造位置 0…N−1。 |
| 22 | <code>    past = (window - 1) // 2</code> | 偶数窗口左侧少一个：16 时 past=7。 |
| 23 | <code>    left = np.maximum(0, target - past)</code> | 窗口左端在回答开头截断。 |
| 24 | <code>    right = np.minimum(len(values), target + window - past)</code> | 半开右端在回答末尾截断，16 时含后 8 个位置。 |
| 25 | <code>    cumulative = np.r_[0., np.cumsum(values)]</code> | 前缀和前补 0，使区间和 O(1)，全向量 O(N)。 |
| 26 | <code>    return (cumulative[right] - cumulative[left]) / (right - left)</code> | 区间和除有效窗口长度；不按片段重置，此行为属于旧 route 定义。 |
| 27 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 28 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 29 | <code>def canonical_delta(values: np.ndarray, units: list[dict]) -&gt; np.ndarray:</code> | canonical_delta 重演历史的单元均值加偏差运算，数学上等于原值，但浮点舍入不必逐位相同。 |
| 30 | <code>    """Preserve the archived unit-mean/deviation floating-point arithmetic."""</code> | canonical_delta 重演历史的单元均值加偏差运算，数学上等于原值，但浮点舍入不必逐位相同。 |
| 31 | <code>    result = np.empty(values.shape, dtype=np.float64)</code> | 为 [N,2] 两种历史对比分配 float64 结果，调用方保证 units 完整覆盖。 |
| 32 | <code>    for unit in units:</code> | 逐原 token 片段取半开切片。 |
| 33 | <code>        region = slice(unit["start"], unit["stop"])</code> | 逐原 token 片段取半开切片。 |
| 34 | <code>        weights = np.ones(unit["stop"] - unit["start"])</code> | 单位权重向量用于保留 archived dot 的运算路径。 |
| 35 | <code>        for column in range(2):</code> | 固定处理 local 和 full 两列，不接受任意列数的通用协议。 |
| 36 | <code>            # The archived vectors were contiguous; BLAS otherwise changes</code> | 注释解释为何先 copy：跨列步长可能使 BLAS 改变求和与一 ULP 结果。 |
| 37 | <code>            # a handful of unit means by one ULP on a strided column.</code> | 注释解释为何先 copy：跨列步长可能使 BLAS 改变求和与一 ULP 结果。 |
| 38 | <code>            part = values[region, column].copy()</code> | 复制当前列使其连续；不只是为了避免写回别名。 |
| 39 | <code>            mean = np.dot(part, weights) / weights.sum()</code> | 按历史 dot/权重总和计算片段均值。 |
| 40 | <code>            result[region, column] = mean + (part - mean)</code> | 先减均值再加回，保留原来的浮点运算顺序，不是新的平滑方法。 |
| 41 | <code>    return result</code> | 返回重演运算后的两列值，尚未过滤 token。 |
| 42 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 43 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 44 | <code>def answer_axis(response: dict, answer_index: int = 0) -&gt; dict:</code> | answer_axis 从编码建立有效 token、回答、片段身份；独立于测量了哪些信号，是所有读出共用的坐标契约。 |
| 45 | <code>    """Original valid-token/unit axes, independent of which signals were measured."""</code> | answer_axis 从编码建立有效 token、回答、片段身份；独立于测量了哪些信号，是所有读出共用的坐标契约。 |
| 46 | <code>    token_id = np.asarray(response["answer_ids"])</code> | 原始 ID 数组，长度是分词后 N，不是有效 token 数。 |
| 47 | <code>    offsets = np.asarray(response["offsets"]).reshape(-1, 2)</code> | offsets 重整为 [N,2]；元素个数不是偶数时 reshape 会先报错。 |
| 48 | <code>    if offsets.shape != (len(token_id), 2):</code> | 检查 token 数与 offset 行数一致。 |
| 49 | <code>        raise ValueError('Response offsets and tokens differ')</code> | 检查 token 数与 offset 行数一致。 |
| 50 | <code>    valid = ((offsets[:, 1] &gt; offsets[:, 0])</code> | 有效 token 必须有正字符长度且不在特殊 ID 集合；不按事实标签、POS 或可疑程度过滤。 |
| 51 | <code>             &amp; ~np.isin(token_id, response["special_ids"]))</code> | 有效 token 必须有正字符长度且不在特殊 ID 集合；不按事实标签、POS 或可疑程度过滤。 |
| 52 | <code>    unit_index = np.empty(len(token_id), dtype=np.int32)</code> | 预分配 int32 片段身份数组，假定 units 覆盖所有 token。 |
| 53 | <code>    for index, unit in enumerate(response["units"]):</code> | 按 units 原顺序赋片段序号；这个函数不验证重叠/缺口，异常手工编码需调用方负责。 |
| 54 | <code>        unit_index[unit["start"]:unit["stop"]] = index</code> | 按 units 原顺序赋片段序号；这个函数不验证重叠/缺口，异常手工编码需调用方负责。 |
| 55 | <code>    count = int(valid.sum())</code> | 统计有效 token 数 M，用于构造回答身份轴。 |
| 56 | <code>    return {</code> | target 是原始 token 下标 [M]，token_id 是对应原 ID；中间被过滤的距离被保留。 |
| 57 | <code>        "target": np.flatnonzero(valid), "token_id": token_id[valid],</code> | target 是原始 token 下标 [M]，token_id 是对应原 ID；中间被过滤的距离被保留。 |
| 58 | <code>        "answer_index": np.full(count, answer_index, dtype=np.int32),</code> | 所有本回答有效 token 共用 answer_index，跨例拼接时必须给不同序号。 |
| 59 | <code>        "unit_index": unit_index[valid],</code> | 仅对有效 token 截取片段身份并返回，不重编号原 target。 |
| 60 | <code>    }</code> | 仅对有效 token 截取片段身份并返回，不重编号原 target。 |
| 61 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 62 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 63 | <code>def answer_rows(observed: dict, response: dict, answer_index: int) -&gt; dict:</code> | answer_rows 将原始观测过滤到有效轴并附元数据；供 API.measure 和离线测量调用。 |
| 64 | <code>    """Align measurements, excluding only empty offsets and special tokens."""</code> | answer_rows 将原始观测过滤到有效轴并附元数据；供 API.measure 和离线测量调用。 |
| 65 | <code>    if not np.array_equal(observed["token_id"], response['answer_ids']):</code> | 观测 token IDs 必须与准备时完整序列逐项一致，否则拒绝，防止缓存/模型输入错位。 |
| 66 | <code>        raise ValueError("Measurement and prepared response token IDs differ")</code> | 观测 token IDs 必须与准备时完整序列逐项一致，否则拒绝，防止缓存/模型输入错位。 |
| 67 | <code>    axis = answer_axis(response, answer_index)</code> | 统一获取原始有效轴，而非每信号各自删行。 |
| 68 | <code>    target = axis['target']</code> | target 是在所有测量信号上使用的同一索引。 |
| 69 | <code>    return {"delta": canonical_delta(observed["delta"], response["units"])[target],</code> | 先在全原始轴作 canonical_delta，然后用 target 过滤，保持 archived 算法定义。 |
| 70 | <code>            "route": window_mean(observed["route"])[target], **axis,</code> | route 也先在原始轴滑窗，再过滤；附 answer/target/token/unit 四条轴。 |
| 71 | <code>            **{key: observed[key][target] for key in DISTRIBUTION_FIELDS if key in observed}}</code> | 只截取实际观测到的可选分布字段；缺失字段不填零。 |
