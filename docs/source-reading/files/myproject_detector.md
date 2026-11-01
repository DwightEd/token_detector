# myproject/detector.py 逐行讲解

基线：`1210120748b3a05bd271bc79debcbed4fdc0c964`；SHA-256：`776fa683830223f1f9393291fa5aa446e6194cb7ab36d5e42f48365e6172ce88`；200 行。

源码与注释逐行并排。一个跨行表达式共用同一操作说明；这不代表各行独立执行。完整调用路径、输入输出轴和理论边界见 [METHOD.md](../METHOD.md)。未执行的 GPU 路径见 [验证边界](../VALIDATION.md)。

| 函数（含嵌套） | 原始行号 | 作用与契约入口 |
|---|---|---|
| `unit_ranges` | 31–36 | unit_ranges 接收两条同长身份轴，返回连续片段 starts/stops；为 shared、sequence、R4 和评价复用。低层空输入会产生 [0,0) 伪段，公共调用者需先处理。 |
| `quantile_rank` | 39–43 | quantile_rank 输入观测和排序参考，输出中位经验 CDF；供 calibrate 使用，保留 ties。 |
| `source_values` | 46–51 | source_values 根据冻结 view 从 delta[N,2] 提取 R4 使用的来源代理。 |
| `tempered_prior` | 54–62 | tempered_prior 对每段计算 log mean exp(tau*d)/tau，再广播回原 token；输入含连续范围和非零冻结温度。 |
| `target_window_mean` | 65–80 | target_window_mean 输入原 token 距离及段范围，输出同轴局部均值；过滤 token 后仍按原 target 距离，而非压缩行号距离选邻居。 |
| `center_units` | 83–90 | center_units 对每个片段的 values 去均值，返回段内零均值创新；供校准后的窗口秩使用。 |
| `raw_features` | 93–102 | raw_features 是 R4.fit/score 共用特征管线，返回 prior、window、starts、stops，避免拟合/推断定义漂移。 |
| `Detector.__init__` | 110–136 | 构造器接收任务及可选参考，并指定分段、创新和校准模式；成功不执行模型前向。 |
| `Detector.calibrate` | 138–144 | calibrate(name,values) 输出参考秩或固定有界对比；两者均不识别事实后验。 |
| `Detector.fit` | 146–158 | fit 只从未标注测量估计参考，修改 self.references 并返回 self。 |
| `Detector.score` | 160–162 | score 是简洁公共接口，仅取 score_components 的 final 向量，诊断计算保持唯一实现。 |
| `Detector.score_components` | 164–182 | score_components 暴露 R4 各阶段供消融，返回全部 token 同轴字典。 |
| `Detector.save` | 184–192 | save 写任务、设置和参考，不写大模型或原始数据。 |
| `Detector.load` | 194–200 | classmethod load 返回构造器验证后的 R4 实例。 |

| 行 | 原始代码 | 操作、目的与边界 |
|---|---|---|
| 1 | <code>"""R4 source calibration, token innovations and bounded continuity."""</code> | 显式历史 R4：参考校准、token 创新与固定连续性；不是当前缺省 shared。 |
| 2 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 3 | <code>from dataclasses import dataclass</code> | dataclass 为冻结任务配方生成 __init__/repr/eq 等方法，字段本身不是学习参数。 |
| 4 | <code>from pathlib import Path</code> | Path 用于 NPZ 参考文件读写。 |
| 5 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 6 | <code>import numpy as np</code> | NumPy 负责历史分位数与确定运算顺序。 |
| 7 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 8 | <code>from .graph import smooth_scores</code> | 复用二状态链消息，不在 Detector 中复制平滑。 |
| 9 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 10 | <code>QUANTILES = np.linspace(0, 1, 10001)</code> | 固定 10001 个 [0,1] 分位点，用于构造历史参考分布。 |
| 11 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 12 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 13 | <code>@dataclass(frozen=True)</code> | Recipe 是 frozen dataclass，生成的初始化接收以下五字段；普通赋值禁止改字段，但它不是深度不可变证明。 |
| 14 | <code>class Recipe:</code> | Recipe 是 frozen dataclass，生成的初始化接收以下五字段；普通赋值禁止改字段，但它不是深度不可变证明。 |
| 15 | <code>    """Task-specific R4 parameters."""</code> | Recipe 是 frozen dataclass，生成的初始化接收以下五字段；普通赋值禁止改字段，但它不是深度不可变证明。 |
| 16 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 17 | <code>    view: str</code> | view 选择完整历史或 local/full 对比均值。 |
| 18 | <code>    tau: float</code> | tau 是片段 log-mean-exp 温度。 |
| 19 | <code>    radius: int</code> | radius 是原 target token 距离半径。 |
| 20 | <code>    innovation_weight: float</code> | innovation_weight 控制片段内中心化局部特征。 |
| 21 | <code>    route_weight: float</code> | route_weight 控制路由代理混入比例。 |
| 22 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 23 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 24 | <code>RECIPES = {</code> | 任务名映射到冻结配方；SharedDetector 只借用这些任务名。 |
| 25 | <code>    "QA": Recipe("pair", .1, 2, .1, .25),</code> | QA 用 pair、tau=.1、半径2、创新.1、路由.25。 |
| 26 | <code>    "Summary": Recipe("full", .25, 4, .25, 0.),</code> | Summary 用 full、tau=.25、半径4、创新.25、无路由权重。 |
| 27 | <code>    "Data2txt": Recipe("pair", .1, 4, .25, 0.),</code> | Data2txt 用 pair、tau=.1、半径4、创新.25、无路由权重。 |
| 28 | <code>}</code> | 结束冻结配方字典，无运行时按标签选参。 |
| 29 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 30 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 31 | <code>def unit_ranges(answer_index: np.ndarray, unit_index: np.ndarray) -&gt; tuple:</code> | unit_ranges 接收两条同长身份轴，返回连续片段 starts/stops；为 shared、sequence、R4 和评价复用。低层空输入会产生 [0,0) 伪段，公共调用者需先处理。 |
| 32 | <code>    """Return contiguous answer/unit ranges on the original packed row order."""</code> | unit_ranges 接收两条同长身份轴，返回连续片段 starts/stops；为 shared、sequence、R4 和评价复用。低层空输入会产生 [0,0) 伪段，公共调用者需先处理。 |
| 33 | <code>    changed = ((answer_index[1:] != answer_index[:-1])</code> | 相邻行 answer 或 unit 任一变化就切断；不对全局相同身份作 groupby 合并。 |
| 34 | <code>               &#124; (unit_index[1:] != unit_index[:-1]))</code> | 相邻行 answer 或 unit 任一变化就切断；不对全局相同身份作 groupby 合并。 |
| 35 | <code>    starts = np.r_[0, np.flatnonzero(changed) + 1]</code> | 首起点固定 0，其余切点取 changed 的下一个位置。 |
| 36 | <code>    return starts, np.r_[starts[1:], len(answer_index)]</code> | 各 stop 是下一 start，最后 stop 是总行数，均半开。 |
| 37 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 38 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 39 | <code>def quantile_rank(values: np.ndarray, reference: np.ndarray) -&gt; np.ndarray:</code> | quantile_rank 输入观测和排序参考，输出中位经验 CDF；供 calibrate 使用，保留 ties。 |
| 40 | <code>    """Historical 10001-quantile mid-CDF, preserving tied values."""</code> | quantile_rank 输入观测和排序参考，输出中位经验 CDF；供 calibrate 使用，保留 ties。 |
| 41 | <code>    left = np.searchsorted(reference, values, side="left")</code> | left 统计严格小于观测的参考项数。 |
| 42 | <code>    right = np.searchsorted(reference, values, side="right")</code> | right 统计小于等于观测的参考项数。 |
| 43 | <code>    return (left + right) / (2 * len(reference))</code> | 两种秩平均再除参考长度，ties 获相同中位秩，不是事实后验。 |
| 44 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 45 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 46 | <code>def source_values(delta: np.ndarray, view: str) -&gt; np.ndarray:</code> | source_values 根据冻结 view 从 delta[N,2] 提取 R4 使用的来源代理。 |
| 47 | <code>    if view == "full":</code> | full 只返回第1列完整回答历史对比，是视图而非保证复制。 |
| 48 | <code>        return delta[:, 1]</code> | full 只返回第1列完整回答历史对比，是视图而非保证复制。 |
| 49 | <code>    if view == "pair":</code> | pair 对 local/full 两列作算术均值，改变观测含义。 |
| 50 | <code>        return .5 * (delta[:, 0] + delta[:, 1])</code> | pair 对 local/full 两列作算术均值，改变观测含义。 |
| 51 | <code>    raise ValueError(f"Unknown source view: {view}")</code> | 未知 view 报错，不默认选择一列。 |
| 52 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 53 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 54 | <code>def tempered_prior(values: np.ndarray, starts: np.ndarray,</code> | tempered_prior 对每段计算 log mean exp(tau*d)/tau，再广播回原 token；输入含连续范围和非零冻结温度。 |
| 55 | <code>                   stops: np.ndarray, tau: float) -&gt; np.ndarray:</code> | tempered_prior 对每段计算 log mean exp(tau*d)/tau，再广播回原 token；输入含连续范围和非零冻结温度。 |
| 56 | <code>    """Broadcast log(mean(exp(tau*delta)))/tau inside each text unit."""</code> | tempered_prior 对每段计算 log mean exp(tau*d)/tau，再广播回原 token；输入含连续范围和非零冻结温度。 |
| 57 | <code>    counts = stops - starts</code> | 每片段有效行数。 |
| 58 | <code>    maxima = np.maximum.reduceat(values, starts)</code> | reduceat 在连续起点计算每段最大 d 作数值锚点。 |
| 59 | <code>    anchors = np.repeat(maxima, counts)</code> | 把段最大值按 counts 广播到 token，避免指数溢出。 |
| 60 | <code>    totals = np.add.reduceat(np.exp(tau * (values - anchors)), starts)</code> | 对 exp(tau*(d−max)) 按段求和。 |
| 61 | <code>    aggregate = maxima + np.log(totals / counts) / tau</code> | 除 counts 取均值、取 log/温度，再补最大值，恢复稳定 log-mean-exp。 |
| 62 | <code>    return np.repeat(aggregate, counts)</code> | 段常数广播到各 token；这个 prior 本身不提供段内区分。 |
| 63 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 64 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 65 | <code>def target_window_mean(values: np.ndarray, target: np.ndarray,</code> | target_window_mean 输入原 token 距离及段范围，输出同轴局部均值；过滤 token 后仍按原 target 距离，而非压缩行号距离选邻居。 |
| 66 | <code>                       starts: np.ndarray, stops: np.ndarray,</code> | target_window_mean 输入原 token 距离及段范围，输出同轴局部均值；过滤 token 后仍按原 target 距离，而非压缩行号距离选邻居。 |
| 67 | <code>                       radius: int) -&gt; np.ndarray:</code> | target_window_mean 输入原 token 距离及段范围，输出同轴局部均值；过滤 token 后仍按原 target 距离，而非压缩行号距离选邻居。 |
| 68 | <code>    """Use raw target distance inside a unit; excluded tokens keep their gaps."""</code> | target_window_mean 输入原 token 距离及段范围，输出同轴局部均值；过滤 token 后仍按原 target 距离，而非压缩行号距离选邻居。 |
| 69 | <code>    result = np.empty_like(values, dtype=np.float64)</code> | 分配 float64 输出以复现历史标度。 |
| 70 | <code>    for start, stop in zip(starts, stops):</code> | 逐片段取其原 target，禁止窗口跨 unit。 |
| 71 | <code>        positions = target[start:stop]</code> | 逐片段取其原 target，禁止窗口跨 unit。 |
| 72 | <code>        left = np.searchsorted(positions, positions - radius, side="left")</code> | 二分找每个位置 target−radius 的包含左界。 |
| 73 | <code>        right = np.searchsorted(positions, positions + radius, side="right")</code> | 二分找 target+radius 的右侧排除界，端点可包含。 |
| 74 | <code>        for row, (first, last) in enumerate(zip(left, right), start=start):</code> | 同时遍历窗口边界与全局输出行号。 |
| 75 | <code>            window = values[start + first:start + last]</code> | 读取当前窗口的 values，不填补被过滤 token 的不存在分数。 |
| 76 | <code>            anchor = window[0]</code> | 以首窗口值作求和锚点。 |
| 77 | <code>            # Keep a fixed addition order for the calibrated reference scale.</code> | 注释声明固定加法顺序是冻结参考标度的数值契约。 |
| 78 | <code>            total = np.cumsum(window - anchor, axis=0)[-1]</code> | 依次累计偏差，而非换用不保证相同舍入的 mean。 |
| 79 | <code>            result[row] = anchor + total / len(window)</code> | 首值加平均偏差得到当前窗口均值。 |
| 80 | <code>    return result</code> | 返回原行序的平滑代理向量。 |
| 81 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 82 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 83 | <code>def center_units(values: np.ndarray, starts: np.ndarray,</code> | center_units 对每个片段的 values 去均值，返回段内零均值创新；供校准后的窗口秩使用。 |
| 84 | <code>                 stops: np.ndarray) -&gt; np.ndarray:</code> | center_units 对每个片段的 values 去均值，返回段内零均值创新；供校准后的窗口秩使用。 |
| 85 | <code>    result = np.empty_like(values, dtype=np.float64)</code> | float64 输出，与原轴同形。 |
| 86 | <code>    for start, stop in zip(starts, stops):</code> | 逐段取视图。 |
| 87 | <code>        unit = values[start:stop]</code> | 逐段取视图。 |
| 88 | <code>        mean = unit[0] + (unit - unit[0]).mean()</code> | 以首项锚定均值运算，保持历史数值路径。 |
| 89 | <code>        result[start:stop] = unit - mean</code> | 将段内偏差写回对应范围，不改变段均值特征本身。 |
| 90 | <code>    return result</code> | 返回中心化向量。 |
| 91 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 92 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 93 | <code>def raw_features(pack: dict, recipe: Recipe, innovation: str = "window") -&gt; tuple:</code> | raw_features 是 R4.fit/score 共用特征管线，返回 prior、window、starts、stops，避免拟合/推断定义漂移。 |
| 94 | <code>    """Compute the source prior and local window used by the readout."""</code> | raw_features 是 R4.fit/score 共用特征管线，返回 prior、window、starts、stops，避免拟合/推断定义漂移。 |
| 95 | <code>    if innovation not in {"window", "point"}:</code> | 创新只支持 window 或 point 两种声明方式。 |
| 96 | <code>        raise ValueError("innovation must be window or point")</code> | 创新只支持 window 或 point 两种声明方式。 |
| 97 | <code>    starts, stops = unit_ranges(pack["answer_index"], pack["unit_index"])</code> | 从 measurement pack 取连续原分段。 |
| 98 | <code>    values = source_values(pack["delta"], recipe.view)</code> | 按任务 view 取来源值。 |
| 99 | <code>    prior = tempered_prior(values, starts, stops, recipe.tau)</code> | 计算每段温度聚合并广播。 |
| 100 | <code>    window = (values.copy() if innovation == "point" else</code> | point 消融复制单 token 值；默认 window 在原 target 距离上求均值。 |
| 101 | <code>              target_window_mean(values, pack["target"], starts, stops, recipe.radius))</code> | point 消融复制单 token 值；默认 window 在原 target 距离上求均值。 |
| 102 | <code>    return prior, window, starts, stops</code> | 将两个特征及边界返回给 FIT/score。 |
| 103 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 104 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 105 | <code>class Detector:</code> | Detector 持有 R4 任务配方与参考分布，提供 fit/score/save/load；由 API 和实验显式选择。 |
| 106 | <code>    """Calibrate and score in-memory measurements for one task."""</code> | Detector 持有 R4 任务配方与参考分布，提供 fit/score/save/load；由 API 和实验显式选择。 |
| 107 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 108 | <code>    method = 'r4'</code> | 方法标识固定 r4。 |
| 109 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 110 | <code>    def __init__(self, task: str, references: dict &#124; None = None, *,</code> | 构造器接收任务及可选参考，并指定分段、创新和校准模式；成功不执行模型前向。 |
| 111 | <code>                 segmentation: str = "punct", innovation: str = "window",</code> | 构造器接收任务及可选参考，并指定分段、创新和校准模式；成功不执行模型前向。 |
| 112 | <code>                 calibration: str = "quantile"):</code> | 构造器接收任务及可选参考，并指定分段、创新和校准模式；成功不执行模型前向。 |
| 113 | <code>        if task not in RECIPES:</code> | 拒绝未知任务。 |
| 114 | <code>            raise ValueError(f"Unknown task: {task}. Choose from {tuple(RECIPES)}")</code> | 拒绝未知任务。 |
| 115 | <code>        self.task = task</code> | 保存任务与相应冻结 Recipe。 |
| 116 | <code>        self.recipe = RECIPES[task]</code> | 保存任务与相应冻结 Recipe。 |
| 117 | <code>        if segmentation not in {"punct", "dependency", "dependency_predictive", "precomputed"}:</code> | 校验分段名；文本 API 会另外排除 precomputed。 |
| 118 | <code>            raise ValueError("Unknown segmentation")</code> | 校验分段名；文本 API 会另外排除 precomputed。 |
| 119 | <code>        if innovation not in {"window", "point"}:</code> | 校验创新定义，避免拟合参考与特征不一致。 |
| 120 | <code>            raise ValueError("innovation must be window or point")</code> | 校验创新定义，避免拟合参考与特征不一致。 |
| 121 | <code>        if calibration not in {"quantile", "none"}:</code> | quantile 用参考 CDF，none 是固定有界变换消融，不等价于默认 shared。 |
| 122 | <code>            raise ValueError("calibration must be quantile or none")</code> | quantile 用参考 CDF，none 是固定有界变换消融，不等价于默认 shared。 |
| 123 | <code>        self.segmentation = segmentation</code> | 保存三个明确设置，序列化时会一起保存。 |
| 124 | <code>        self.innovation = innovation</code> | 保存三个明确设置，序列化时会一起保存。 |
| 125 | <code>        self.calibration = calibration</code> | 保存三个明确设置，序列化时会一起保存。 |
| 126 | <code>        if calibration == "quantile" and references is not None:</code> | 只有 quantile 且提供参考时检查其内容；None 代表尚未 FIT。 |
| 127 | <code>            if set(references) != {"prior", "window", "route"}:</code> | 参考必须恰好包含 prior/window/route，即使任务路由权重为0也保留冻结 schema。 |
| 128 | <code>                raise ValueError("References require prior, window and route arrays")</code> | 参考必须恰好包含 prior/window/route，即使任务路由权重为0也保留冻结 schema。 |
| 129 | <code>            references = {name: np.asarray(value, dtype=np.float64)</code> | 各参考转 float64，不在构造器重新估计。 |
| 130 | <code>                          for name, value in references.items()}</code> | 各参考转 float64，不在构造器重新估计。 |
| 131 | <code>            if any(values.ndim != 1 or not len(values) or not np.isfinite(values).all()</code> | 必须一维、非空、有限且已排序；允许 ties。 |
| 132 | <code>                   or np.any(np.diff(values) &lt; 0) for values in references.values()):</code> | 必须一维、非空、有限且已排序；允许 ties。 |
| 133 | <code>                raise ValueError("Reference arrays must be finite, sorted and nonempty")</code> | 必须一维、非空、有限且已排序；允许 ties。 |
| 134 | <code>        if calibration == "none" and references:</code> | none 模式拒绝非空参考，防止把固定 sigmoid 说成参考校准。 |
| 135 | <code>            raise ValueError("A no-FIT readout cannot use reference arrays")</code> | none 模式拒绝非空参考，防止把固定 sigmoid 说成参考校准。 |
| 136 | <code>        self.references = {} if calibration == "none" else references</code> | none 存空字典以区别未 FIT 的 None；quantile 保留已验证参考或 None。 |
| 137 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 138 | <code>    def calibrate(self, name: str, values: np.ndarray) -&gt; np.ndarray:</code> | calibrate(name,values) 输出参考秩或固定有界对比；两者均不识别事实后验。 |
| 139 | <code>        """Reference rank or a fixed bounded contrast; neither is a truth posterior."""</code> | calibrate(name,values) 输出参考秩或固定有界对比；两者均不识别事实后验。 |
| 140 | <code>        if self.calibration == "quantile":</code> | quantile 查询对应参考的 mid-CDF。 |
| 141 | <code>            return quantile_rank(values, self.references[name])</code> | quantile 查询对应参考的 mid-CDF。 |
| 142 | <code>        if name == "route":</code> | none 的 route 从近似 [-1,1] 线性映到 [0,1] 再裁剪。 |
| 143 | <code>            return np.clip(.5 * (np.asarray(values) + 1.), 0., 1.)</code> | none 的 route 从近似 [-1,1] 线性映到 [0,1] 再裁剪。 |
| 144 | <code>        return np.exp(-np.logaddexp(0., -np.asarray(values)))</code> | none 的来源/窗口值用稳定 sigmoid 压到 [0,1]。 |
| 145 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 146 | <code>    def fit(self, pack: dict):</code> | fit 只从未标注测量估计参考，修改 self.references 并返回 self。 |
| 147 | <code>        """Fit references from the supplied unlabeled training measurements."""</code> | fit 只从未标注测量估计参考，修改 self.references 并返回 self。 |
| 148 | <code>        if not len(pack["target"]):</code> | 至少有一个有效 token 才能定义经验分位数。 |
| 149 | <code>            raise ValueError("At least one training token is required")</code> | 至少有一个有效 token 才能定义经验分位数。 |
| 150 | <code>        if self.calibration == "none":</code> | none 不需要参考，但仍先检查输入非空，然后直接返回。 |
| 151 | <code>            return self</code> | none 不需要参考，但仍先检查输入非空，然后直接返回。 |
| 152 | <code>        prior, window, _, _ = raw_features(pack, self.recipe, self.innovation)</code> | 复用与推断相同的 prior/window 特征定义。 |
| 153 | <code>        self.references = {</code> | 分别计算 prior、window、route 的固定10001分位点，不读取标签或训练神经网络。 |
| 154 | <code>            "prior": np.quantile(prior, QUANTILES),</code> | 分别计算 prior、window、route 的固定10001分位点，不读取标签或训练神经网络。 |
| 155 | <code>            "window": np.quantile(window, QUANTILES),</code> | 分别计算 prior、window、route 的固定10001分位点，不读取标签或训练神经网络。 |
| 156 | <code>            "route": np.quantile(pack["route"], QUANTILES),</code> | 分别计算 prior、window、route 的固定10001分位点，不读取标签或训练神经网络。 |
| 157 | <code>        }</code> | 分别计算 prior、window、route 的固定10001分位点，不读取标签或训练神经网络。 |
| 158 | <code>        return self</code> | 返回已更新参考的同一对象。 |
| 159 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 160 | <code>    def score(self, pack: dict) -&gt; np.ndarray:</code> | score 是简洁公共接口，仅取 score_components 的 final 向量，诊断计算保持唯一实现。 |
| 161 | <code>        """Return final per-token risk scores without logging or saving features."""</code> | score 是简洁公共接口，仅取 score_components 的 final 向量，诊断计算保持唯一实现。 |
| 162 | <code>        return self.score_components(pack)['final']</code> | score 是简洁公共接口，仅取 score_components 的 final 向量，诊断计算保持唯一实现。 |
| 163 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 164 | <code>    def score_components(self, pack: dict) -&gt; dict:</code> | score_components 暴露 R4 各阶段供消融，返回全部 token 同轴字典。 |
| 165 | <code>        """The actual R4 computation, exposed for diagnostic and ablation callers."""</code> | score_components 暴露 R4 各阶段供消融，返回全部 token 同轴字典。 |
| 166 | <code>        if self.references is None:</code> | 未 FIT 的 quantile 模式拒绝打分。 |
| 167 | <code>            raise ValueError("Fit or load reference distributions before scoring")</code> | 未 FIT 的 quantile 模式拒绝打分。 |
| 168 | <code>        if not len(pack["target"]):</code> | 空有效轴返回所有诊断的空 float64 数组，不启动底层 reduceat/链。 |
| 169 | <code>            return {k: np.empty(0, dtype=np.float64) for k in</code> | 空有效轴返回所有诊断的空 float64 数组，不启动底层 reduceat/链。 |
| 170 | <code>                    ['point', 'prior', 'source', 'window', 'window_rank', 'innovation', 'base', 'unary', 'final']}</code> | 空有效轴返回所有诊断的空 float64 数组，不启动底层 reduceat/链。 |
| 171 | <code>        prior, window, starts, stops = raw_features(pack, self.recipe, self.innovation)</code> | 共用原始特征管线。 |
| 172 | <code>        source = self.calibrate("prior", prior)</code> | 段 prior 经参考/固定校准得到 source。 |
| 173 | <code>        route = self.calibrate("route", pack["route"])</code> | route 同样校准，即使当前任务权重为0仍计算，保留历史行为。 |
| 174 | <code>        rank = self.calibrate("window", window)</code> | 窗口特征经相应参考校准为秩。 |
| 175 | <code>        innovation = center_units(rank, starts, stops)</code> | 段内去均值得 token 创新，不让整段偏移重复进入。 |
| 176 | <code>        recipe = self.recipe</code> | 取当前任务冻结权重。 |
| 177 | <code>        base = (1 - recipe.route_weight) * source + recipe.route_weight * route</code> | 将 source 与 route 按固定凸组合形成 base。 |
| 178 | <code>        unary = base + recipe.innovation_weight * innovation</code> | 加创新项形成 unary，它不保证在 [0,1] 内。 |
| 179 | <code>        return {'point': source_values(pack['delta'], recipe.view),</code> | 保存用于归因的原始 point 来源代理。 |
| 180 | <code>                'prior': prior, 'source': source, 'window': window, 'window_rank': rank,</code> | 返回聚合、校准、创新与融合中间值，均是诊断，不独立训练。 |
| 181 | <code>                'innovation': innovation, 'base': base, 'unary': unary,</code> | 返回聚合、校准、创新与融合中间值，均是诊断，不独立训练。 |
| 182 | <code>                'final': smooth_scores(unary, pack['target'], pack['answer_index'])}</code> | final 对 unary 在原回答 target 上做固定链连续性，允许跨 unit 但不能跨 answer。 |
| 183 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 184 | <code>    def save(self, path: Path) -&gt; None:</code> | save 写任务、设置和参考，不写大模型或原始数据。 |
| 185 | <code>        """Save only the task and calibrated reference distributions."""</code> | save 写任务、设置和参考，不写大模型或原始数据。 |
| 186 | <code>        if self.references is None:</code> | 未 FIT 的参考不能保存成伪可用模型。 |
| 187 | <code>            raise ValueError("Fit reference distributions before saving")</code> | 未 FIT 的参考不能保存成伪可用模型。 |
| 188 | <code>        with Path(path).open("wb") as stream:</code> | 用文件句柄精确写到指定路径。 |
| 189 | <code>            np.savez_compressed(stream, task=np.asarray(self.task),</code> | NPZ 保存 task、segmentation、innovation、calibration 和参考数组，none 模式没有参考键。 |
| 190 | <code>                                segmentation=np.asarray(self.segmentation),</code> | NPZ 保存 task、segmentation、innovation、calibration 和参考数组，none 模式没有参考键。 |
| 191 | <code>                                innovation=np.asarray(self.innovation),</code> | NPZ 保存 task、segmentation、innovation、calibration 和参考数组，none 模式没有参考键。 |
| 192 | <code>                                calibration=np.asarray(self.calibration), **self.references)</code> | NPZ 保存 task、segmentation、innovation、calibration 和参考数组，none 模式没有参考键。 |
| 193 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 194 | <code>    @classmethod</code> | classmethod load 返回构造器验证后的 R4 实例。 |
| 195 | <code>    def load(cls, path: Path):</code> | classmethod load 返回构造器验证后的 R4 实例。 |
| 196 | <code>        with np.load(path, allow_pickle=False) as saved:</code> | allow_pickle=False，避免依赖对象反序列化。 |
| 197 | <code>            settings = {key: str(saved[key]) for key in</code> | 可选设置存在才恢复，旧文件缺这些字段时使用构造器历史默认。 |
| 198 | <code>                        ("segmentation", "innovation", "calibration") if key in saved}</code> | 可选设置存在才恢复，旧文件缺这些字段时使用构造器历史默认。 |
| 199 | <code>            references = {key: saved[key] for key in ("prior", "window", "route") if key in saved}</code> | 取实际存在的三个参考数组；字段不完整由构造器拒绝。 |
| 200 | <code>            return cls(str(saved["task"]), references, **settings)</code> | 以任务、设置、参考统一构造，关闭 NPZ 后数组仍由实例持有。 |
