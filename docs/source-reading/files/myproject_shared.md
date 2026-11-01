# myproject/shared.py 逐行讲解

基线：`1210120748b3a05bd271bc79debcbed4fdc0c964`；SHA-256：`180a4b1d02b94ea560aa12c9b661ed1439ae0f9416e0613c1d3b3573b82889f5`；105 行。

源码与注释逐行并排。一个跨行表达式共用同一操作说明；这不代表各行独立执行。完整调用路径、输入输出轴和理论边界见 [METHOD.md](../METHOD.md)。未执行的 GPU 路径见 [验证边界](../VALIDATION.md)。

| 函数（含嵌套） | 原始行号 | 作用与契约入口 |
|---|---|---|
| `SharedDetector.__init__` | 31–39 | 构造器默认 QA、标点分段和 shared；不读取数据、不拟合先验。 |
| `SharedDetector.score` | 41–42 | score 是公共简洁接口：接收 measurement pack，仅返回 score 向量；诊断仍在 score_components。 |
| `SharedDetector.score_components` | 44–59 | score_components 是文本、缓存与消融的唯一调度点，返回分数和对应诊断字典。 |
| `SharedDetector.score_full_gap` | 61–69 | score_full_gap 供缓存 full-gap 直接调用，输入三条等长轴，仅返回分数；不会伪造 local/route 数据。 |
| `SharedDetector._score_values` | 71–92 | _score_values 处理空输入、内核选择及额外输出；私有调用约定 axes 与 values 已对齐。 |
| `SharedDetector.save` | 94–97 | save(path) 仅持久化配置；公共 API.save 委托这里。 |
| `SharedDetector.load` | 99–105 | classmethod load 以 cls 构造兼容子类，返回已验证配置实例。 |

| 行 | 原始代码 | 操作、目的与边界 |
|---|---|---|
| 1 | <code>"""Primary token-preserving, fixed-prior source-expert readout."""</code> | 固定先验专家读出模块；它不训练事实分类器。 |
| 2 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 3 | <code>from pathlib import Path</code> | Path 仅用于保存/读取配置路径。 |
| 4 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 5 | <code>import numpy as np</code> | NumPy 管理 token 数组和无 pickle 的 NPZ。 |
| 6 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 7 | <code>from .detector import RECIPES</code> | 复用任务名称集合；默认共享方法并不使用 R4 的任务特定系数。 |
| 8 | <code>from .pooling import grouped_scores</code> | 引入交换型 Beta 片段推断，是 shared 默认的数值内核。 |
| 9 | <code>from .sequence import grouped_persistent_scores, responsibility_logits</code> | 引入持续性积分内核及 witness 内专家责任转换，避免在 API/实验处复制实现。 |
| 10 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 11 | <code>SEQUENCE_METHODS = ('persistent', 'witness', 'responsibility', 'conditional', 'onset',</code> | 显式列出可选择的持续性/诊断方法名；这些名字表示不同模型，不能视为等价算法。 |
| 12 | <code>                    'token_state', 'adaptive_token', 'proof_revision',</code> | 显式列出可选择的持续性/诊断方法名；这些名字表示不同模型，不能视为等价算法。 |
| 13 | <code>                    'symmetric_local', 'matched_local', 'fixed_local', 'fixed_persistence')</code> | 显式列出可选择的持续性/诊断方法名；这些名字表示不同模型，不能视为等价算法。 |
| 14 | <code>WITNESS_METHODS = ('witness', 'responsibility')</code> | 这两个方法都先采用 witness 混合发射；responsibility 还要换最终输出含义。 |
| 15 | <code>LOCAL_METHODS = {'symmetric_local': 'symmetric', 'matched_local': 'matched',</code> | 把四种局部消融映射到 sequence 中相应的 prevalence/persistence 先验设置。 |
| 16 | <code>                 'fixed_local': 'fixed', 'fixed_persistence': 'fixed_persistence'}</code> | 把四种局部消融映射到 sequence 中相应的 prevalence/persistence 先验设置。 |
| 17 | <code>SHARED_METHODS = ('shared', *SEQUENCE_METHODS)</code> | 所有共享家族由默认 shared 和显式候选组成。 |
| 18 | <code>METHODS = (*SHARED_METHODS, 'r4')</code> | CLI/API 接受的方法集合再加入历史 r4，避免隐式切换旧方法。 |
| 19 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 20 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 21 | <code>class SharedDetector:</code> | SharedDetector 持有 task/segmentation/method 配置，数值推断委托给唯一内核；由 HallucinationDetector 和实验调用。 |
| 22 | <code>    """Share a latent tendency within a span while retaining each token's full gap.</code> | 类文档说明共享的是专家倾向，proof_revision 的有界诊断概率也不是校准后的事实概率。 |
| 23 | <code> </code> | 类文档说明共享的是专家倾向，proof_revision 的有界诊断概率也不是校准后的事实概率。 |
| 24 | <code>    Tasks share fixed priors and the same full-history source contrast.</code> | 类文档说明共享的是专家倾向，proof_revision 的有界诊断概率也不是校准后的事实概率。 |
| 25 | <code>    Scores are expert log odds, or bounded diagnostic probabilities for proof_revision;</code> | 类文档说明共享的是专家倾向，proof_revision 的有界诊断概率也不是校准后的事实概率。 |
| 26 | <code>    neither is a calibrated factuality probability.</code> | 类文档说明共享的是专家倾向，proof_revision 的有界诊断概率也不是校准后的事实概率。 |
| 27 | <code>    """</code> | 类文档说明共享的是专家倾向，proof_revision 的有界诊断概率也不是校准后的事实概率。 |
| 28 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 29 | <code>    method = 'shared'</code> | 类级默认方法标识；实例构造时会被 self.method 覆盖。 |
| 30 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 31 | <code>    def __init__(self, task='QA', *, segmentation='punct', method='shared'):</code> | 构造器默认 QA、标点分段和 shared；不读取数据、不拟合先验。 |
| 32 | <code>        if task not in RECIPES:</code> | 未知任务立即拒绝，统一公共 API 的任务命名。 |
| 33 | <code>            raise ValueError(f'Unknown task: {task}')</code> | 未知任务立即拒绝，统一公共 API 的任务命名。 |
| 34 | <code>        if segmentation not in {'punct', 'dependency', 'dependency_predictive', 'precomputed'}:</code> | 允许四种分段配置，其中 precomputed 只适合测量包，文本 API 会另行拒绝。 |
| 35 | <code>            raise ValueError('Unknown segmentation')</code> | 允许四种分段配置，其中 precomputed 只适合测量包，文本 API 会另行拒绝。 |
| 36 | <code>        if method not in SHARED_METHODS:</code> | 该类只接受共享家族方法，不能错误装载 r4。 |
| 37 | <code>            raise ValueError(f'Unknown shared method: {method}')</code> | 该类只接受共享家族方法，不能错误装载 r4。 |
| 38 | <code>        self.task, self.segmentation = task, segmentation</code> | 保存任务与分段配置，不保存模型或 token 状态。 |
| 39 | <code>        self.method = method</code> | 保存显式选择的方法以决定后续读出。 |
| 40 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 41 | <code>    def score(self, pack):</code> | score 是公共简洁接口：接收 measurement pack，仅返回 score 向量；诊断仍在 score_components。 |
| 42 | <code>        return self.score_components(pack)['score']</code> | score 是公共简洁接口：接收 measurement pack，仅返回 score 向量；诊断仍在 score_components。 |
| 43 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 44 | <code>    def score_components(self, pack):</code> | score_components 是文本、缓存与消融的唯一调度点，返回分数和对应诊断字典。 |
| 45 | <code>        """One implementation behind text prediction, cached runs and ablations."""</code> | score_components 是文本、缓存与消融的唯一调度点，返回分数和对应诊断字典。 |
| 46 | <code>        if self.method == 'proof_revision':</code> | proof_revision 分支延迟加载额外来源证明修订器；该机制不属于默认 shared。 |
| 47 | <code>            from .revision import revise_scores</code> | proof_revision 分支延迟加载额外来源证明修订器；该机制不属于默认 shared。 |
| 48 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 49 | <code>            proofs = pack.get('proofs', {'components': [], 'diagnostics': {'reason': 'missing_source_context'}})</code> | 无 proof 上下文时使用空组件并明确 missing_source_context 原因，不捏造事实证明。 |
| 50 | <code>            result = revise_scores(pack['delta'][:, 1], pack['answer_index'], pack['unit_index'],</code> | 修订器接收完整历史列 delta[:,1]、原始分段轴与 proof 组件。 |
| 51 | <code>                                   proofs['components'])</code> | 修订器接收完整历史列 delta[:,1]、原始分段轴与 proof 组件。 |
| 52 | <code>            result['proof_diagnostics'] = {'compiler': proofs['diagnostics'],</code> | 将编译与修订诊断聚合到 proof_diagnostics，同时移走内部 components 键。 |
| 53 | <code>                                            'revision': result.pop('components')}</code> | 将编译与修订诊断聚合到 proof_diagnostics，同时移走内部 components 键。 |
| 54 | <code>            return result</code> | 提前返回 proof 结果，不再进入普通专家模型。 |
| 55 | <code>        if not len(pack['target']):</code> | 没有有效 target 时走空数组分支，避免底层要求非空片段的 DP。 |
| 56 | <code>            return self._score_values(np.empty(0), None)</code> | 没有有效 target 时走空数组分支，避免底层要求非空片段的 DP。 |
| 57 | <code>        values = (pack['calibrated_gap'] if self.method == 'conditional'</code> | conditional 需要全词表构造的 calibrated_gap；其他普通分支仅取 full gap，未使用 local/route。 |
| 58 | <code>                  else pack['delta'][:, 1])</code> | conditional 需要全词表构造的 calibrated_gap；其他普通分支仅取 full gap，未使用 local/route。 |
| 59 | <code>        return self._score_values(values, (pack['answer_index'], pack['unit_index']))</code> | 把观测与两条原始分组轴交给统一内核包装。 |
| 60 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 61 | <code>    def score_full_gap(self, values, answer_index, unit_index):</code> | score_full_gap 供缓存 full-gap 直接调用，输入三条等长轴，仅返回分数；不会伪造 local/route 数据。 |
| 62 | <code>        """Score actual full-history contrasts without inventing local/route fields."""</code> | score_full_gap 供缓存 full-gap 直接调用，输入三条等长轴，仅返回分数；不会伪造 local/route 数据。 |
| 63 | <code>        if self.method in {'conditional', 'proof_revision'}:</code> | conditional 和 proof_revision 依赖额外观测，只有 full gap 时必须拒绝。 |
| 64 | <code>            raise ValueError('This method requires observations beyond a full source gap')</code> | conditional 和 proof_revision 依赖额外观测，只有 full gap 时必须拒绝。 |
| 65 | <code>        values, answers, units = map(np.asarray, (values, answer_index, unit_index))</code> | 将三个输入转换为数组，保持 token 顺序。 |
| 66 | <code>        if (values.ndim != 1 or answers.shape != values.shape or units.shape != values.shape</code> | 要求有限一维 values 且两个身份轴同形；形状相同仍不证明身份语义正确。 |
| 67 | <code>                or not np.isfinite(values).all()):</code> | 要求有限一维 values 且两个身份轴同形；形状相同仍不证明身份语义正确。 |
| 68 | <code>            raise ValueError('Matching finite full-gap and original token axes required')</code> | 要求有限一维 values 且两个身份轴同形；形状相同仍不证明身份语义正确。 |
| 69 | <code>        return self._score_values(values, (answers, units))['score']</code> | 复用相同调度返回 score，保证文本/缓存方法一致。 |
| 70 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 71 | <code>    def _score_values(self, values, axes):</code> | _score_values 处理空输入、内核选择及额外输出；私有调用约定 axes 与 values 已对齐。 |
| 72 | <code>        if not len(values):</code> | 空输入返回独立空分数，并把 values 统一为空向量供后续诊断计算。 |
| 73 | <code>            values, result = np.empty(0), {'score': np.empty(0)}</code> | 空输入返回独立空分数，并把 values 统一为空向量供后续诊断计算。 |
| 74 | <code>        else:</code> | 非空输入才启动状态积分。 |
| 75 | <code>            result = (grouped_scores(values, *axes, alpha=1., beta=1.) if self.method == 'shared'</code> | shared 固定 Beta(1,1)，调用 grouped_scores；这里不会使用 fit_prior。 |
| 76 | <code>                      else grouped_persistent_scores(values, *axes,</code> | 其他普通方法转到 grouped_persistent_scores，在原片段边界重置状态。 |
| 77 | <code>                                                     witness=self.method in WITNESS_METHODS or self.method in LOCAL_METHODS,</code> | witness/responsibility 及四个 local 消融开启 witness 发射。 |
| 78 | <code>                                                     onset=self.method == 'onset',</code> | 只有 onset 输出入口状态模型。 |
| 79 | <code>                                                     token_state=self.method == 'token_state',</code> | 只有 token_state 开启固定噪声通道分解。 |
| 80 | <code>                                                     adaptive=self.method == 'adaptive_token',</code> | 只有 adaptive_token 比较两种声明结构。 |
| 81 | <code>                                                     prevalence=LOCAL_METHODS.get(self.method, 'integrated')))</code> | local 方法使用对应固定/对称先验，其余默认积分 prevalence。 |
| 82 | <code>        if self.method in WITNESS_METHODS:</code> | witness 与 responsibility 都保留两个层次的专家诊断。 |
| 83 | <code>            result['witness_score'] = result['score']</code> | 原状态分数保存为 witness_score，供比较而非丢失。 |
| 84 | <code>            result['responsibility_score'] = responsibility_logits(values, result['score'])</code> | 将区域状态 odds 和原 d 转为 absent 专家责任 odds。 |
| 85 | <code>            if self.method == 'responsibility':</code> | 仅 responsibility 方法把转换后的责任分数设为最终输出；witness 保留原状态分数。 |
| 86 | <code>                result['score'] = result['responsibility_score']</code> | 仅 responsibility 方法把转换后的责任分数设为最终输出；witness 保留原状态分数。 |
| 87 | <code>        if self.method == 'onset' and not len(values):</code> | 空 onset 也提供同名空 entry_score，保持返回 schema。 |
| 88 | <code>            result['entry_score'] = np.empty(0)</code> | 空 onset 也提供同名空 entry_score，保持返回 schema。 |
| 89 | <code>        if self.method == 'adaptive_token' and not len(values):</code> | 空 adaptive 同样提供四个诊断字段，不伪造非空结果。 |
| 90 | <code>            result.update({key: np.empty(0) for key in ['witness_score', 'token_state_score',</code> | 空 adaptive 同样提供四个诊断字段，不伪造非空结果。 |
| 91 | <code>                           'uniform_model_score', 'deviation_probability']})</code> | 空 adaptive 同样提供四个诊断字段，不伪造非空结果。 |
| 92 | <code>        return result</code> | 返回本次独立结果，不把跨回答消息留在对象中。 |
| 93 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 94 | <code>    def save(self, path: Path):</code> | save(path) 仅持久化配置；公共 API.save 委托这里。 |
| 95 | <code>        with Path(path).open('wb') as stream:</code> | 二进制文件对象确保按用户给定文件名写，不隐式追加 .npz。 |
| 96 | <code>            np.savez_compressed(stream, method=np.asarray(self.method), task=np.asarray(self.task),</code> | 压缩保存 method/task/segmentation 三个无对象数组，没有模型权重或训练样本。 |
| 97 | <code>                                segmentation=np.asarray(self.segmentation))</code> | 压缩保存 method/task/segmentation 三个无对象数组，没有模型权重或训练样本。 |
| 98 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 99 | <code>    @classmethod</code> | classmethod load 以 cls 构造兼容子类，返回已验证配置实例。 |
| 100 | <code>    def load(cls, path: Path):</code> | classmethod load 以 cls 构造兼容子类，返回已验证配置实例。 |
| 101 | <code>        with np.load(path, allow_pickle=False) as saved:</code> | 禁用 pickle 读取 NPZ，并在 with 结束关闭资源。 |
| 102 | <code>            method = str(saved['method'])</code> | 将 NPZ 标量转换回 Python 方法字符串。 |
| 103 | <code>            if method not in SHARED_METHODS:</code> | 拒绝非共享家族配置，防止把 R4 references 当共享模型装载。 |
| 104 | <code>                raise ValueError('Not a shared detector configuration')</code> | 拒绝非共享家族配置，防止把 R4 references 当共享模型装载。 |
| 105 | <code>            return cls(str(saved['task']), segmentation=str(saved['segmentation']), method=method)</code> | 交回构造器统一校验任务/分段/方法，未执行任何模型推断。 |
