# myproject/attention.py 逐行讲解

基线：`1210120748b3a05bd271bc79debcbed4fdc0c964`；SHA-256：`4ec72d0ec9de72f6883fd9463df078b2e4c0c6cdf6cc9d531ceb1551d0e0c4b8`；56 行。

源码与注释逐行并排。一个跨行表达式共用同一操作说明；这不代表各行独立执行。完整调用路径、输入输出轴和理论边界见 [METHOD.md](../METHOD.md)。未执行的 GPU 路径见 [验证边界](../VALIDATION.md)。

| 函数（含嵌套） | 原始行号 | 作用与契约入口 |
|---|---|---|
| `attention_profiles` | 9–56 | no_grad 的 attention_profiles 返回 [N,P+N+1] 两个 head 奇偶剖面及mask；当前行是实际 y_t 的 P+t 状态，区别于预测 y_t 的 P+t−1。供非标点分段调用。 |
| `attention_profiles.record` | 25–38 | 嵌套 record 捕获当前块 first/last 和 result；hooks 在同一循环迭代内执行并移除，避免晚绑定跨块。 |

| 行 | 原始代码 | 操作、目的与边界 |
|---|---|---|
| 1 | <code>"""Original-prompt, post-token attention routing; no graph walks or retokenization."""</code> | 原提示、post-token attention 诊断；不做图游走、不重分词。 |
| 2 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 3 | <code>import numpy as np</code> | NumPy 用于CPU诊断，PyTorch用于设备上 attention 归约。 |
| 4 | <code>import torch</code> | NumPy 用于CPU诊断，PyTorch用于设备上 attention 归约。 |
| 5 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 6 | <code>from myproject.model import QUERY_CHUNK, attention_backend, input_ids, prefill</code> | 复用模型分块、后端恢复、ID张量和KV prefill，避免另一套前向实现。 |
| 7 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 8 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 9 | <code>@torch.no_grad()</code> | no_grad 的 attention_profiles 返回 [N,P+N+1] 两个 head 奇偶剖面及mask；当前行是实际 y_t 的 P+t 状态，区别于预测 y_t 的 P+t−1。供非标点分段调用。 |
| 10 | <code>def attention_profiles(model, prompt, answer):</code> | no_grad 的 attention_profiles 返回 [N,P+N+1] 两个 head 奇偶剖面及mask；当前行是实际 y_t 的 P+t 状态，区别于预测 y_t 的 P+t−1。供非标点分段调用。 |
| 11 | <code>    """Keep prompt identities, past answer identities and one self coordinate.</code> | no_grad 的 attention_profiles 返回 [N,P+N+1] 两个 head 奇偶剖面及mask；当前行是实际 y_t 的 P+t 状态，区别于预测 y_t 的 P+t−1。供非标点分段调用。 |
| 12 | <code> </code> | no_grad 的 attention_profiles 返回 [N,P+N+1] 两个 head 奇偶剖面及mask；当前行是实际 y_t 的 P+t 状态，区别于预测 y_t 的 P+t−1。供非标点分段调用。 |
| 13 | <code>    Returned head-parity views are correlated diagnostics. Segmentation rows P+t</code> | no_grad 的 attention_profiles 返回 [N,P+N+1] 两个 head 奇偶剖面及mask；当前行是实际 y_t 的 P+t 状态，区别于预测 y_t 的 P+t−1。供非标点分段调用。 |
| 14 | <code>    belong to actual y_t states, distinct from P+t-1 token probability rows.</code> | no_grad 的 attention_profiles 返回 [N,P+N+1] 两个 head 奇偶剖面及mask；当前行是实际 y_t 的 P+t 状态，区别于预测 y_t 的 P+t−1。供非标点分段调用。 |
| 15 | <code>    """</code> | no_grad 的 attention_profiles 返回 [N,P+N+1] 两个 head 奇偶剖面及mask；当前行是实际 y_t 的 P+t 状态，区别于预测 y_t 的 P+t−1。供非标点分段调用。 |
| 16 | <code>    if not prompt or not answer:</code> | 提示/回答都必须非空，避免无定义的行归一与前向。 |
| 17 | <code>        raise ValueError('Prompt and answer must be nonempty')</code> | 提示/回答都必须非空，避免无定义的行归一与前向。 |
| 18 | <code>    p, n = len(prompt), len(answer)</code> | p 是完整提示长度，n 是原始回答 token 数。 |
| 19 | <code>    result = torch.zeros((2, n, p + n + 1), device=model.device, dtype=torch.float32)</code> | 分配 FP32 [2,n,p+n+1] 累积表，目的地坐标为原提示、原回答历史及统一 self 槽；内存随完整回答平方增长。 |
| 20 | <code>    cache = prefill(model, prompt)</code> | 这里 prefill 全部 prompt，不留最后提示 token，因为随后直接输入 y_t 来读 post-token 状态。 |
| 21 | <code>    for first in range(0, n, QUERY_CHUNK):</code> | 按 QUERY_CHUNK 遍历回答，设本块半开结束并准备临时 hooks。 |
| 22 | <code>        last = min(first + QUERY_CHUNK, n)</code> | 按 QUERY_CHUNK 遍历回答，设本块半开结束并准备临时 hooks。 |
| 23 | <code>        handles = []</code> | 按 QUERY_CHUNK 遍历回答，设本块半开结束并准备临时 hooks。 |
| 24 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 25 | <code>        def record(_module, _args, output):</code> | 嵌套 record 捕获当前块 first/last 和 result；hooks 在同一循环迭代内执行并移除，避免晚绑定跨块。 |
| 26 | <code>            attention = output[1][0].detach().float()</code> | 取 batch0 attention[H,Q,K]，detach 后转 FP32。 |
| 27 | <code>            if attention.shape[0] &lt; 2:</code> | 至少两个头才可形成两个非空奇偶视图。 |
| 28 | <code>                raise ValueError('Two head-parity views require at least two heads')</code> | 至少两个头才可形成两个非空奇偶视图。 |
| 29 | <code>            for group in [0, 1]:</code> | 分别对偶/奇头求均值得 [Q,K]，是相关诊断视图而非独立样本拆分。 |
| 30 | <code>                weights = attention[group::2].mean(0)</code> | 分别对偶/奇头求均值得 [Q,K]，是相关诊断视图而非独立样本拆分。 |
| 31 | <code>                result[group, first:last, :p] += weights[:, :p]</code> | 完整提示 key 坐标逐项保留，不把 query/source提示先合并。 |
| 32 | <code>                history = weights[:, p:p + last].clone()</code> | 复制当前可见回答 key 部分，防止之后mask原地修改模型 attention。 |
| 33 | <code>                rows = torch.arange(last - first, device=model.device)</code> | 取每个当前回答 token 看到自身 key 的 attention，另存 self 槽。 |
| 34 | <code>                self_weights = history[rows, torch.arange(first, last, device=model.device)].clone()</code> | 取每个当前回答 token 看到自身 key 的 attention，另存 self 槽。 |
| 35 | <code>                history *= (torch.arange(last, device=model.device)[None, :]</code> | 只允许回答 key 下标严格小于当前回答位置，排除自身及未来；本块多行也各自使用不同因果支持。 |
| 36 | <code>                            &lt; torch.arange(first, last, device=model.device)[:, None])</code> | 只允许回答 key 下标严格小于当前回答位置，排除自身及未来；本块多行也各自使用不同因果支持。 |
| 37 | <code>                result[group, first:last, p:p + last] += history</code> | 过去回答 token 保留独立原始 key 坐标。 |
| 38 | <code>                result[group, first:last, -1] += self_weights</code> | 每行 self 质量归并最后一列，不与历史 token 质量混合。 |
| 39 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 40 | <code>        try:</code> | 在 try 内逐层注册相同归约 hook，handles 记录用于清理。 |
| 41 | <code>            for layer in model.model.layers:</code> | 在 try 内逐层注册相同归约 hook，handles 记录用于清理。 |
| 42 | <code>                handles.append(layer.self_attn.register_forward_hook(record))</code> | 在 try 内逐层注册相同归约 hook，handles 记录用于清理。 |
| 43 | <code>            with attention_backend(model, 'eager'):</code> | eager 前向直接输入当前回答块，复用完整提示 cache；为读实际 token 后状态而非预测查询。 |
| 44 | <code>                model.model(input_ids=input_ids(model, answer[first:last]),</code> | eager 前向直接输入当前回答块，复用完整提示 cache；为读实际 token 后状态而非预测查询。 |
| 45 | <code>                            past_key_values=cache, use_cache=True, return_dict=True)</code> | eager 前向直接输入当前回答块，复用完整提示 cache；为读实际 token 后状态而非预测查询。 |
| 46 | <code>        finally:</code> | 任何前向/注册异常都移除已注册 hooks，避免污染后续模型调用。 |
| 47 | <code>            for handle in handles:</code> | 任何前向/注册异常都移除已注册 hooks，避免污染后续模型调用。 |
| 48 | <code>                handle.remove()</code> | 任何前向/注册异常都移除已注册 hooks，避免污染后续模型调用。 |
| 49 | <code>    values = result.cpu().numpy().astype(np.float64) / len(model.model.layers)</code> | 跨层总质量除层数，CPU上转 float64进行后续归一。 |
| 50 | <code>    # BF16 attention rounding can change the sum slightly; retain raw masses too.</code> | 保留两视图每行 raw_mass，记录 BF16 attention 舍入而非假定严格和1。 |
| 51 | <code>    mass = values.sum(-1)</code> | 保留两视图每行 raw_mass，记录 BF16 attention 舍入而非假定严格和1。 |
| 52 | <code>    values /= mass[:, :, None]</code> | 按每视图每行质量归一，供分段概率模型使用。 |
| 53 | <code>    available = np.ones((n, p + n + 1), bool)</code> | 初始全部目的地可用，提示及统一self始终允许。 |
| 54 | <code>    available[:, p:p+n] = np.tril(np.ones((n, n), bool), -1)</code> | 回答历史块只允许严格下三角，当前/未来位置不可用。 |
| 55 | <code>    return {'fit': values[0], 'hold': values[1], 'available': available, 'raw_mass': mass,</code> | 返回 fit/hold、同一个因果mask、原质量、提示长度和原回答ID，保持可追踪身份。 |
| 56 | <code>            'prompt_tokens': p, 'answer_ids': np.asarray(answer)}</code> | 返回 fit/hold、同一个因果mask、原质量、提示长度和原回答ID，保持可追踪身份。 |
