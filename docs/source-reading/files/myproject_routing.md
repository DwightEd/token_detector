# myproject/routing.py 逐行讲解

基线：`1210120748b3a05bd271bc79debcbed4fdc0c964`；SHA-256：`dd12983cfaa1a5c888c29158e582dc1bdb4206868ceec5e694565e132b6f5fee`；68 行。

源码与注释逐行并排。一个跨行表达式共用同一操作说明；这不代表各行独立执行。完整调用路径、输入输出轴和理论边界见 [METHOD.md](../METHOD.md)。未执行的 GPU 路径见 [验证边界](../VALIDATION.md)。

| 函数（含嵌套） | 原始行号 | 作用与契约入口 |
|---|---|---|
| `projection_grams` | 8–17 | projection_grams(model) 返回逐层 [H,d_h,d_h] Gram 列表，供 Observer 缓存复用。 |
| `append_value_energy` | 20–25 | append_value_energy 接收 values[H,K,d_h]、Gram 和旧 energy[H,K_old]，只计算新增 key 的能量并返回扩展结果。 |
| `route_magnitudes` | 28–42 | route_magnitudes 输入 attention[Q,H,K] 与 energy[H,K]，返回 [Q,H] 的来源/历史/总幅值；调用者再跨头/层聚合。 |
| `observe_route` | 45–68 | observe_route 是临时 hook 上下文：输入 cache、跨块 energy 状态等，yield 本块逐层归约字典。异常退出仍移除已注册 hooks。 |
| `observe_route.observe` | 51–60 | 嵌套 observe(index) 是工厂，将当前层号绑定在独立闭包，避免循环索引晚绑定。 |
| `observe_route.observe.record` | 52–59 | 嵌套 record 是真实 forward hook，输入 module/args/output，只产生侧面统计而不替换模型输出。 |

| 行 | 原始代码 | 操作、目的与边界 |
|---|---|---|
| 1 | <code>"""Physical-head W_O V magnitudes and prediction-row source/history routing."""</code> | 计算各物理注意力头投影 value 的范数和来源/历史路由；这不是图游走或因果归因。 |
| 2 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 3 | <code>from contextlib import contextmanager</code> | contextmanager 管理临时 hooks 的安装与移除。 |
| 4 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 5 | <code>import torch</code> | PyTorch 负责设备上矩阵范数与 forward hooks。 |
| 6 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 7 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 8 | <code>def projection_grams(model):</code> | projection_grams(model) 返回逐层 [H,d_h,d_h] Gram 列表，供 Observer 缓存复用。 |
| 9 | <code>    grams = []</code> | 每层独立存一份投影度量。 |
| 10 | <code>    for layer in model.model.layers:</code> | 沿 Llama 风格 model.model.layers 定位 self_attn。 |
| 11 | <code>        attention = layer.self_attn</code> | 沿 Llama 风格 model.model.layers 定位 self_attn。 |
| 12 | <code>        heads = attention.q_proj.out_features // attention.head_dim</code> | 从 q_proj 输出宽度除 head_dim 得物理 query 头数 H；GQA 的 KV 头数可能不同。 |
| 13 | <code>        projection = attention.o_proj.weight.detach().float()</code> | 取 o_proj 权重 [D,H*d_h]，detach 后转 FP32，保持在当前设备。 |
| 14 | <code>        matrices = projection.reshape(projection.shape[0], heads, attention.head_dim)</code> | 把列按头切成 [D,H,d_h]，对应每个头输出到残差空间的线性投影。 |
| 15 | <code>        matrices = matrices.permute(1, 2, 0)</code> | 重排为 [H,d_h,D]，使 value 行向量右乘投影。 |
| 16 | <code>        grams.append(matrices @ matrices.transpose(-1, -2))</code> | 每头乘自身转置得 [H,d_h,d_h]，以后用 v G vᵀ 计算投影后范数平方。 |
| 17 | <code>    return grams</code> | 返回与层索引一一对应的 Gram 列表。 |
| 18 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 19 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 20 | <code>def append_value_energy(values, gram, previous):</code> | append_value_energy 接收 values[H,K,d_h]、Gram 和旧 energy[H,K_old]，只计算新增 key 的能量并返回扩展结果。 |
| 21 | <code>    """Reuse old keys' W_O V energies exactly as in the original routing collector."""</code> | append_value_energy 接收 values[H,K,d_h]、Gram 和旧 energy[H,K_old]，只计算新增 key 的能量并返回扩展结果。 |
| 22 | <code>    start = 0 if previous is None else previous.shape[-1]</code> | 已有缓存长度决定新增 key 起点；首次从 0 开始。 |
| 23 | <code>    unseen = values[:, start:]</code> | 切出 [H,K−K_old,d_h] 的未计算 values，避免重复投影旧 key。 |
| 24 | <code>    energy = ((unseen @ gram) * unseen).sum(-1).clamp_min(0)</code> | 先算 vG 再逐维乘 v 并求和得 ||W_O,h v||²；clamp_min(0) 消除舍入带来的微小负值。 |
| 25 | <code>    return energy if previous is None else torch.cat((previous, energy), dim=-1)</code> | 首次直接返回，否则沿 key 轴连接；不沿 token 查询轴缓存。 |
| 26 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 27 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 28 | <code>def route_magnitudes(attention, energy, prompt_length, source_mask, first_query):</code> | route_magnitudes 输入 attention[Q,H,K] 与 energy[H,K]，返回 [Q,H] 的来源/历史/总幅值；调用者再跨头/层聚合。 |
| 29 | <code>    """Retain each physical head's &#124;&#124;W_O,h V_h,k&#124;&#124; before aggregating routing."""</code> | route_magnitudes 输入 attention[Q,H,K] 与 energy[H,K]，返回 [Q,H] 的来源/历史/总幅值；调用者再跨头/层聚合。 |
| 30 | <code>    magnitude = attention * energy.sqrt()</code> | sqrt energy 得每 key 的投影 value 范数，经广播乘注意力概率；先求范数再加，不能称投影和的范数。 |
| 31 | <code>    source = torch.zeros(attention.shape[-1], device=attention.device)</code> | 构造 key 轴上的来源选择向量，新增回答 key 默认不属于来源。 |
| 32 | <code>    source[:prompt_length] = torch.tensor(source_mask, device=attention.device)</code> | 提示前 P 项填来源 mask，和 tokenizer 的删除区间一致。 |
| 33 | <code>    history = (torch.arange(attention.shape[-1], device=attention.device) &gt;= prompt_length).float()</code> | key≥P 的位置标成回答历史；首个预测查询仍在提示末尾。 |
| 34 | <code>    source_norm = magnitude @ source</code> | 沿 key 轴加权求和，得到各查询/头的来源幅值。 |
| 35 | <code>    history_norm = magnitude @ history</code> | 同样聚合回答历史幅值，不包括提示的 query 文本。 |
| 36 | <code>    if first_query and source_mask[-1]:</code> | 首预测行若最后提示 token 被标为来源，去掉该行的自身位置贡献，保持历史 collector 定义。 |
| 37 | <code>        source_norm[0] -= magnitude[0, :, prompt_length - 1]</code> | 首预测行若最后提示 token 被标为来源，去掉该行的自身位置贡献，保持历史 collector 定义。 |
| 38 | <code>    return dict(</code> | 转 CPU NumPy，返回来源、历史和所有 key 的幅值；总幅值包括其他提示 token，三者不是互斥概率。 |
| 39 | <code>        norm_source=source_norm.cpu().numpy(),</code> | 转 CPU NumPy，返回来源、历史和所有 key 的幅值；总幅值包括其他提示 token，三者不是互斥概率。 |
| 40 | <code>        norm_history=history_norm.cpu().numpy(),</code> | 转 CPU NumPy，返回来源、历史和所有 key 的幅值；总幅值包括其他提示 token，三者不是互斥概率。 |
| 41 | <code>        norm_total=magnitude.sum(-1).cpu().numpy(),</code> | 转 CPU NumPy，返回来源、历史和所有 key 的幅值；总幅值包括其他提示 token，三者不是互斥概率。 |
| 42 | <code>    )</code> | 转 CPU NumPy，返回来源、历史和所有 key 的幅值；总幅值包括其他提示 token，三者不是互斥概率。 |
| 43 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 44 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 45 | <code>@contextmanager</code> | observe_route 是临时 hook 上下文：输入 cache、跨块 energy 状态等，yield 本块逐层归约字典。异常退出仍移除已注册 hooks。 |
| 46 | <code>def observe_route(model, cache, grams, energies, prompt_length, source_mask, start):</code> | observe_route 是临时 hook 上下文：输入 cache、跨块 energy 状态等，yield 本块逐层归约字典。异常退出仍移除已注册 hooks。 |
| 47 | <code>    """Reduce each layer's current attention block; never retain all-layer edges."""</code> | observe_route 是临时 hook 上下文：输入 cache、跨块 energy 状态等，yield 本块逐层归约字典。异常退出仍移除已注册 hooks。 |
| 48 | <code>    layers = {}</code> | 本块 layers 由 hooks 填充，不保存所有层 attention 边。 |
| 49 | <code>    handles = []</code> | 保存 hook handle 以便退出移除。 |
| 50 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 51 | <code>    def observe(index):</code> | 嵌套 observe(index) 是工厂，将当前层号绑定在独立闭包，避免循环索引晚绑定。 |
| 52 | <code>        def record(module, args, output):</code> | 嵌套 record 是真实 forward hook，输入 module/args/output，只产生侧面统计而不替换模型输出。 |
| 53 | <code>            attention = output[1][0].detach().float().transpose(0, 1)</code> | 从 eager attention 输出取 batch0，[H,Q,K] 转 FP32 后换为 [Q,H,K]。 |
| 54 | <code>            values = cache.layers[index].values[0].float()</code> | 取当前层 KV cache 的 batch0 value，[H_KV,K,d_h]。 |
| 55 | <code>            values = values.repeat_interleave(attention.shape[1] // values.shape[0], dim=0)</code> | GQA 时按组复制 KV 头至 query 物理头 H；这依赖标准头分组顺序。 |
| 56 | <code>            energies[index] = append_value_energy(values, grams[index], energies.get(index))</code> | 按层扩展旧 key 能量缓存，副作用写入本次 source_full 的 energies 字典。 |
| 57 | <code>            layers[index] = route_magnitudes(</code> | 对当前层做 source/history/total 归约，首块 start==0 决定首查询自位置修正。 |
| 58 | <code>                attention, energies[index], prompt_length, source_mask, start == 0,</code> | 对当前层做 source/history/total 归约，首块 start==0 决定首查询自位置修正。 |
| 59 | <code>            )</code> | 对当前层做 source/history/total 归约，首块 start==0 决定首查询自位置修正。 |
| 60 | <code>        return record</code> | 把绑定层号的 record 返回给 register_forward_hook。 |
| 61 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 62 | <code>    for index, layer in enumerate(model.model.layers):</code> | 逐层注册 self_attn hook，并保存 handles；注册发生在 try 前，极端注册中途失败没有统一清理分支。 |
| 63 | <code>        handles.append(layer.self_attn.register_forward_hook(observe(index)))</code> | 逐层注册 self_attn hook，并保存 handles；注册发生在 try 前，极端注册中途失败没有统一清理分支。 |
| 64 | <code>    try:</code> | yield 空字典引用给 with 体，前向发生后该字典被 hooks 原地填满。 |
| 65 | <code>        yield layers</code> | yield 空字典引用给 with 体，前向发生后该字典被 hooks 原地填满。 |
| 66 | <code>    finally:</code> | with 体成功或异常都逐 handle.remove，避免后续前向重复收集或占内存。 |
| 67 | <code>        for handle in handles:</code> | with 体成功或异常都逐 handle.remove，避免后续前向重复收集或占内存。 |
| 68 | <code>            handle.remove()</code> | with 体成功或异常都逐 handle.remove，避免后续前向重复收集或占内存。 |
