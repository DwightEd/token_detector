# myproject/source.py 逐行讲解

基线：`1210120748b3a05bd271bc79debcbed4fdc0c964`；SHA-256：`838476d6af7889d22efc09925693245218663f04faf0f79bb9474ef923e1ecbc`；54 行。

源码与注释逐行并排。一个跨行表达式共用同一操作说明；这不代表各行独立执行。完整调用路径、输入输出轴和理论边界见 [METHOD.md](../METHOD.md)。未执行的 GPU 路径见 [验证边界](../VALIDATION.md)。

| 函数（含嵌套） | 原始行号 | 作用与契约入口 |
|---|---|---|
| `source_full` | 15–38 | source_full 输入已 prefill 的 cache 和来源 mask/Gram，返回长度 N 的完整 logp 与 route；调用者保证模型冻结/no_grad。 |
| `source_condition` | 41–54 | source_condition 在一个来源条件下输出 local/full/route；提示前缀共用一份 prefill，回答历史在每局部片段重置。 |

| 行 | 原始代码 | 操作、目的与边界 |
|---|---|---|
| 1 | <code>"""Present/absent source likelihoods with full and unit-local histories."""</code> | 来源有/无与回答完整/片段局部历史的测量调度。 |
| 2 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 3 | <code>import numpy as np</code> | NumPy 拼接每块观测与层/头路由统计。 |
| 4 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 5 | <code>from .model import (</code> | 复用模型模块的分块尺寸、后端上下文、输入张量、KV prefill 与似然实现，避免重复加载/归一化逻辑。 |
| 6 | <code>    QUERY_CHUNK,</code> | 复用模型模块的分块尺寸、后端上下文、输入张量、KV prefill 与似然实现，避免重复加载/归一化逻辑。 |
| 7 | <code>    attention_backend,</code> | 复用模型模块的分块尺寸、后端上下文、输入张量、KV prefill 与似然实现，避免重复加载/归一化逻辑。 |
| 8 | <code>    input_ids,</code> | 复用模型模块的分块尺寸、后端上下文、输入张量、KV prefill 与似然实现，避免重复加载/归一化逻辑。 |
| 9 | <code>    prefill,</code> | 复用模型模块的分块尺寸、后端上下文、输入张量、KV prefill 与似然实现，避免重复加载/归一化逻辑。 |
| 10 | <code>    score_continuation,</code> | 复用模型模块的分块尺寸、后端上下文、输入张量、KV prefill 与似然实现，避免重复加载/归一化逻辑。 |
| 11 | <code>)</code> | 复用模型模块的分块尺寸、后端上下文、输入张量、KV prefill 与似然实现，避免重复加载/归一化逻辑。 |
| 12 | <code>from .routing import observe_route</code> | 路由 hooks 只在 source-present 完整历史前向中收集；默认 shared 分数不使用 route。 |
| 13 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 14 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 15 | <code>def source_full(model, cache, prompt, answer, source_mask, grams, on_logits=None):</code> | source_full 输入已 prefill 的 cache 和来源 mask/Gram，返回长度 N 的完整 logp 与 route；调用者保证模型冻结/no_grad。 |
| 16 | <code>    queries = [prompt[-1], *answer[:-1]]</code> | 与 SDPA scorer 相同的右移预测查询，当前位置不偷看目标 y_t。 |
| 17 | <code>    likelihoods = []</code> | 收集本块实际 token 的 logp。 |
| 18 | <code>    routes = []</code> | 收集本块逐 token 的路由标量。 |
| 19 | <code>    energies = {}</code> | 每层历史 key 的 value 能量缓存仅属于本次回答，不跨回答复用。 |
| 20 | <code>    with attention_backend(model, "eager"):</code> | 切换 eager 以使 attention 权重可被 hook 读取；因此与 absent 的 SDPA 存在数值后端差异。 |
| 21 | <code>        for start in range(0, len(answer), QUERY_CHUNK):</code> | 按最多 QUERY_CHUNK=16 个预测位置遍历。 |
| 22 | <code>            with observe_route(model, cache, grams, energies, len(prompt), source_mask, start) as layers:</code> | 为当前块注册临时逐层 hooks；layers 在前向后填充，并在离开时移除 hooks。 |
| 23 | <code>                hidden = model.model(</code> | 主干以 [1,C] 右移查询和同一 cache 前向，取 batch 0 的 [C,D] 隐藏态；cache 随预测增长。 |
| 24 | <code>                    input_ids=input_ids(model, queries[start:start + QUERY_CHUNK]),</code> | 主干以 [1,C] 右移查询和同一 cache 前向，取 batch 0 的 [C,D] 隐藏态；cache 随预测增长。 |
| 25 | <code>                    past_key_values=cache, use_cache=True, return_dict=True,</code> | 主干以 [1,C] 右移查询和同一 cache 前向，取 batch 0 的 [C,D] 隐藏态；cache 随预测增长。 |
| 26 | <code>                ).last_hidden_state[0]</code> | 主干以 [1,C] 右移查询和同一 cache 前向，取 batch 0 的 [C,D] 隐藏态；cache 随预测增长。 |
| 27 | <code>            native = model.lm_head(hidden)</code> | 原生 lm_head 投影到 [C,V] 词表 logits。 |
| 28 | <code>            if on_logits is not None:</code> | 可选回调保留全词表来源分布，conditional 方法需要它。 |
| 29 | <code>                on_logits(start, native)</code> | 可选回调保留全词表来源分布，conditional 方法需要它。 |
| 30 | <code>            logits = native.float().log_softmax(-1)</code> | 转 FP32 后沿词表作 log_softmax；保留历史实现的运算顺序。 |
| 31 | <code>            targets = input_ids(model, answer[start:start + QUERY_CHUNK])[0]</code> | 构造当前块实际回答 token IDs，形状 [C]。 |
| 32 | <code>            likelihoods.append(logits.gather(1, targets[:, None])[:, 0].cpu().numpy())</code> | 按词表轴收集实际 token logp，移至 CPU，为完整回答拼接。 |
| 33 | <code>            source = np.stack([layers[index]["norm_source"] for index in layers], axis=1)</code> | 把各层来源幅值沿层轴堆叠成 [C,L,H]，不是注意力边图。 |
| 34 | <code>            history = np.stack([layers[index]["norm_history"] for index in layers], axis=1)</code> | 同样得到历史回答路由的 [C,L,H]。 |
| 35 | <code>            total = np.stack([layers[index]["norm_total"] for index in layers], axis=1)</code> | 得到所有 key 总幅值 [C,L,H] 作为分母尺度。 |
| 36 | <code>            ratio = (history - source).sum(-1) / np.maximum(total.sum(-1), 1e-12)</code> | 先在头轴求和，计算每层 (历史幅值−来源幅值)/总幅值，分母至少 1e−12。 |
| 37 | <code>            routes.append(ratio.mean(-1))</code> | 再沿层轴平均得到 [C] 路由分数；这是范数加权路由代理，不是有符号因果贡献。 |
| 38 | <code>    return np.concatenate(likelihoods), np.concatenate(routes)</code> | 两个返回数组都沿原始 N 个回答 token 排列。 |
| 39 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 40 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 41 | <code>def source_condition(model, prompt, answer, units, source_mask=None, grams=None, on_logits=None):</code> | source_condition 在一个来源条件下输出 local/full/route；提示前缀共用一份 prefill，回答历史在每局部片段重置。 |
| 42 | <code>    """Full history and independent unit-local histories share one prompt prefill."""</code> | source_condition 在一个来源条件下输出 local/full/route；提示前缀共用一份 prefill，回答历史在每局部片段重置。 |
| 43 | <code>    cache = prefill(model, prompt[:-1])</code> | 只 prefill prompt[:-1]，把最后提示 token 留给首个预测查询，避免重复缓存。 |
| 44 | <code>    if source_mask is None:</code> | 没有 source_mask 时只用 SDPA 计算完整 logp，route 明确为 None；用于 absent 条件。 |
| 45 | <code>        full = score_continuation(model, cache, prompt[-1], answer, on_logits)</code> | 没有 source_mask 时只用 SDPA 计算完整 logp，route 明确为 None；用于 absent 条件。 |
| 46 | <code>        route = None</code> | 没有 source_mask 时只用 SDPA 计算完整 logp，route 明确为 None；用于 absent 条件。 |
| 47 | <code>    else:</code> | 有 source_mask 时走 eager 路由收集版本，并传入预计算 Gram。 |
| 48 | <code>        full, route = source_full(model, cache, prompt, answer, source_mask, grams, on_logits)</code> | 有 source_mask 时走 eager 路由收集版本，并传入预计算 Gram。 |
| 49 | <code>    local = np.empty_like(full)</code> | 分配与 full 同 dtype/长度的局部历史 logp。 |
| 50 | <code>    for unit in units:</code> | 按已定义的半开片段区间独立测量。 |
| 51 | <code>        cache.crop(len(prompt) - 1)</code> | 将 cache 裁剪回提示的 P−1 项，丢弃上次回答 KV；不是磁盘清缓存。 |
| 52 | <code>        start, stop = unit["start"], unit["stop"]</code> | 读取原回答 token 的 [start,stop) 边界。 |
| 53 | <code>        local[start:stop] = score_continuation(model, cache, prompt[-1], answer[start:stop])</code> | 局部 scorer 只见提示加当前片段的先前 token，结果写回原始位置；不调用全词表回调。 |
| 54 | <code>    return local, full, route</code> | 返回同轴 local/full 及可选 route；函数本身仍不计算检测分数。 |
