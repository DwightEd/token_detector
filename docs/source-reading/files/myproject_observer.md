# myproject/observer.py 逐行讲解

基线：`1210120748b3a05bd271bc79debcbed4fdc0c964`；SHA-256：`054d7baee1089872b5cbd081d03e6b146fe40309807854dca33dabc7f71da747`；39 行。

源码与注释逐行并排。一个跨行表达式共用同一操作说明；这不代表各行独立执行。完整调用路径、输入输出轴和理论边界见 [METHOD.md](../METHOD.md)。未执行的 GPU 路径见 [验证边界](../VALIDATION.md)。

| 函数（含嵌套） | 原始行号 | 作用与契约入口 |
|---|---|---|
| `Observer.__init__` | 14–16 | 构造器接收已加载模型；投影 Gram 要求 Llama 风格层/头结构。 |
| `Observer.measure` | 18–39 | no_grad 装饰整次观测，避免 CPU numpy 转换遇到梯度图并控制推断内存。 |

| 行 | 原始代码 | 操作、目的与边界 |
|---|---|---|
| 1 | <code>"""Native source deletion and prediction-row message measurements."""</code> | 原生来源删除和预测位置路由的观测层，负责观测，不负责分类。 |
| 2 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 3 | <code>import numpy as np</code> | NumPy 构造 float64 差值矩阵与 token 身份数组。 |
| 4 | <code>import torch</code> | PyTorch no_grad 保护实际模型前向。 |
| 5 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 6 | <code>from .routing import projection_grams</code> | projection_grams 预计算各层物理头输出投影的 Gram 以复用。 |
| 7 | <code>from .source import source_condition</code> | source_condition 统一执行每种来源世界的 full/local 测量。 |
| 8 | <code>from .distribution import SourceDistributions</code> | SourceDistributions 仅在 conditional 方法需要完整词表诊断时使用。 |
| 9 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 10 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 11 | <code>class Observer:</code> | Observer 由 API 惰性创建，持有冻结模型及一次性 Gram，不存跨回答后验状态。 |
| 12 | <code>    """Keep one frozen model and reuse output-projection Gram matrices."""</code> | Observer 由 API 惰性创建，持有冻结模型及一次性 Gram，不存跨回答后验状态。 |
| 13 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 14 | <code>    def __init__(self, model):</code> | 构造器接收已加载模型；投影 Gram 要求 Llama 风格层/头结构。 |
| 15 | <code>        self.model = model</code> | 保存模型引用，不复制大模型。 |
| 16 | <code>        self.grams = projection_grams(model)</code> | 一次性计算所有层 W_O 的 head Gram；即使默认 shared 不使用 route 也会计算这项。 |
| 17 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 18 | <code>    @torch.no_grad()</code> | no_grad 装饰整次观测，避免 CPU numpy 转换遇到梯度图并控制推断内存。 |
| 19 | <code>    def measure(self, source: dict, response: dict, *, full_distribution=False) -&gt; dict:</code> | measure 输入 encode_source/encode_response 的字典，输出 delta[N,2]、route[N]、token_id[N] 和可选分布诊断。 |
| 20 | <code>        """Measure source-present/absent × full/local histories for actual tokens."""</code> | measure 输入 encode_source/encode_response 的字典，输出 delta[N,2]、route[N]、token_id[N] 和可选分布诊断。 |
| 21 | <code>        prompt = source["prompt_with_source"]</code> | 完整提示 token 序列为 present 世界的输入。 |
| 22 | <code>        answer = response["answer_ids"]</code> | 固定回答 token 序列，两个世界都测同一 y，不重新生成。 |
| 23 | <code>        distributions = SourceDistributions(answer) if full_distribution else None</code> | 只有 full_distribution=True 才分配完整词表收集器，普通方法节省此部分内存。 |
| 24 | <code>        local, full, route = source_condition(</code> | present 世界测量 full/local/route，回调按块收集 present logits；传来源 mask 开启 eager hooks。 |
| 25 | <code>            self.model, prompt, answer, response["units"], source["source_mask"], self.grams,</code> | present 世界测量 full/local/route，回调按块收集 present logits；传来源 mask 开启 eager hooks。 |
| 26 | <code>            on_logits=distributions.present if distributions else None,</code> | present 世界测量 full/local/route，回调按块收集 present logits；传来源 mask 开启 eager hooks。 |
| 27 | <code>        )</code> | present 世界测量 full/local/route，回调按块收集 present logits；传来源 mask 开启 eager hooks。 |
| 28 | <code>        absent_local, absent_full, _ = source_condition(</code> | absent 世界使用删来源提示而保留回答与片段轴，不传 mask/Gram；full 回调配对 present 块。 |
| 29 | <code>            self.model, source["prompt_without_source"], answer, response["units"],</code> | absent 世界使用删来源提示而保留回答与片段轴，不传 mask/Gram；full 回调配对 present 块。 |
| 30 | <code>            on_logits=distributions.absent if distributions else None,</code> | absent 世界使用删来源提示而保留回答与片段轴，不传 mask/Gram；full 回调配对 present 块。 |
| 31 | <code>        )</code> | absent 世界使用删来源提示而保留回答与片段轴，不传 mask/Gram；full 回调配对 present 块。 |
| 32 | <code>        result = {</code> | 组装无 I/O 的原始观测结果。 |
| 33 | <code>            "delta": np.column_stack((absent_local.astype(np.float64) - local,</code> | 第 0 列是 absent_local−present_local；先转 absent 为 float64，避免两 FP32 数先相减丢精度。 |
| 34 | <code>                                      absent_full.astype(np.float64) - full)),</code> | 第 1 列是 absent_full−present_full，默认 shared 只读取这一列。 |
| 35 | <code>            "route": route, "token_id": np.asarray(answer, dtype=np.int64),</code> | 附 present 路由和原始 int64 token 身份，便于之后严格轴对齐。 |
| 36 | <code>        }</code> | 结束基础结果字典，尚未过滤特殊 token。 |
| 37 | <code>        if distributions is not None:</code> | 有词表收集器才检查所有块完整并并入条件预测诊断；否则不造零占位字段。 |
| 38 | <code>            result.update(distributions.finish())</code> | 有词表收集器才检查所有块完整并并入条件预测诊断；否则不造零占位字段。 |
| 39 | <code>        return result</code> | 把观测交给 answer_rows 过滤/对齐，未进行标签读取或事实校准。 |
