# myproject/model.py 逐行讲解

基线：`1210120748b3a05bd271bc79debcbed4fdc0c964`；SHA-256：`1e7393ab159f4e9e9feab91c5fe073499a90f31cddb07f2aee8eec18385d2264`；68 行。

源码与注释逐行并排。一个跨行表达式共用同一操作说明；这不代表各行独立执行。完整调用路径、输入输出轴和理论边界见 [METHOD.md](../METHOD.md)。未执行的 GPU 路径见 [验证边界](../VALIDATION.md)。

| 函数（含嵌套） | 原始行号 | 作用与契约入口 |
|---|---|---|
| `load_model` | 15–22 | load_model(path,device) 返回冻结 eval 模型；由 API.from_pretrained 调用，加载失败向上传播。 |
| `input_ids` | 25–26 | input_ids(model,tokens) 将一维 token 序列变成设备上的 [1,L] int64 tensor；批量维固定为 1。 |
| `attention_backend` | 29–37 | attention_backend 是异常安全上下文管理器；yield 前切后端，离开时恢复。由 prefill、似然与 attention 收集共用。 |
| `prefill` | 40–48 | prefill(model,tokens) 创建并返回已扩展的 KV cache；只计算主干，不计算全部词表 logits。 |
| `score_continuation` | 51–68 | score_continuation 输入已有前缀 cache、最后提示 token、回答及可选 logits 回调；返回 [N] CPU logp，原地追加 KV。 |

| 行 | 原始代码 | 操作、目的与边界 |
|---|---|---|
| 1 | <code>"""Frozen Llama loading, KV prefill and token likelihoods."""</code> | 冻结模型加载与 KV cache/teacher-forcing 似然；这里没有优化器或训练循环。 |
| 2 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 3 | <code>from contextlib import contextmanager</code> | contextmanager 让生成器负责 attention backend 的进入与恢复。 |
| 4 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 5 | <code>import numpy as np</code> | NumPy 仅拼接已经移到 CPU 的逐 token 似然。 |
| 6 | <code>import torch</code> | PyTorch 提供模型设备、tensor、dtype 和后端设置。 |
| 7 | <code>from transformers import AutoModelForCausalLM</code> | AutoModelForCausalLM 从本地配置选择模型架构；实测适配还依赖 Llama 风格 model.model/lm_head 接口。 |
| 8 | <code>from transformers.cache_utils import DynamicCache</code> | DynamicCache 保存逐层键值；这是推断 KV cache，不是磁盘检查点或响应缓存。 |
| 9 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 10 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 11 | <code>PREFILL_CHUNK = 256</code> | 每次 prefill 至多 256 个输入 token，限制单次前向大小，不缩短总上下文。 |
| 12 | <code>QUERY_CHUNK = 16</code> | 每次查询最多 16 个预测位置；最终分数仍逐原始 token 输出。 |
| 13 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 14 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 15 | <code>def load_model(path, device):</code> | load_model(path,device) 返回冻结 eval 模型；由 API.from_pretrained 调用，加载失败向上传播。 |
| 16 | <code>    """Load the source-observation model with its original native BF16 arithmetic."""</code> | load_model(path,device) 返回冻结 eval 模型；由 API.from_pretrained 调用，加载失败向上传播。 |
| 17 | <code>    torch.backends.cuda.matmul.allow_tf32 = False</code> | 关闭 CUDA matmul/cuDNN TF32，修改进程全局标志，函数不会恢复；不会据此证明使用某个 GPU kernel。 |
| 18 | <code>    torch.backends.cudnn.allow_tf32 = False</code> | 关闭 CUDA matmul/cuDNN TF32，修改进程全局标志，函数不会恢复；不会据此证明使用某个 GPU kernel。 |
| 19 | <code>    model = AutoModelForCausalLM.from_pretrained(</code> | 仅本地加载，指定 BF16 和 SDPA，缺文件不自动联网补齐。 |
| 20 | <code>        path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True,</code> | 仅本地加载，指定 BF16 和 SDPA，缺文件不自动联网补齐。 |
| 21 | <code>    ).to(device).eval().requires_grad_(False)</code> | 移动到设备，eval 关闭训练态行为，requires_grad_(False) 冻结所有参数；仍需调用方 no_grad 避免输入图。 |
| 22 | <code>    return model</code> | 返回模型对象，不返回 tokenizer。 |
| 23 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 24 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 25 | <code>def input_ids(model, tokens):</code> | input_ids(model,tokens) 将一维 token 序列变成设备上的 [1,L] int64 tensor；批量维固定为 1。 |
| 26 | <code>    return torch.tensor([tokens], device=model.device, dtype=torch.long)</code> | input_ids(model,tokens) 将一维 token 序列变成设备上的 [1,L] int64 tensor；批量维固定为 1。 |
| 27 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 28 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 29 | <code>@contextmanager</code> | attention_backend 是异常安全上下文管理器；yield 前切后端，离开时恢复。由 prefill、似然与 attention 收集共用。 |
| 30 | <code>def attention_backend(model, implementation):</code> | attention_backend 是异常安全上下文管理器；yield 前切后端，离开时恢复。由 prefill、似然与 attention 收集共用。 |
| 31 | <code>    """Restore the caller's attention backend even when a native forward fails."""</code> | attention_backend 是异常安全上下文管理器；yield 前切后端，离开时恢复。由 prefill、似然与 attention 收集共用。 |
| 32 | <code>    previous = model.config._attn_implementation</code> | 记录调用者原 attention 实现配置。 |
| 33 | <code>    model.set_attn_implementation(implementation)</code> | 切到请求后端；若这一步失败尚未进入 try，不会执行后续恢复。 |
| 34 | <code>    try:</code> | 将上下文控制权交还 with 体，在调用方执行真正前向。 |
| 35 | <code>        yield</code> | 将上下文控制权交还 with 体，在调用方执行真正前向。 |
| 36 | <code>    finally:</code> | 无论 with 体成功或异常都恢复原实现，防止 eager 泄漏到之后的 SDPA 测量。 |
| 37 | <code>        model.set_attn_implementation(previous)</code> | 无论 with 体成功或异常都恢复原实现，防止 eager 泄漏到之后的 SDPA 测量。 |
| 38 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 39 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 40 | <code>def prefill(model, tokens):</code> | prefill(model,tokens) 创建并返回已扩展的 KV cache；只计算主干，不计算全部词表 logits。 |
| 41 | <code>    cache = DynamicCache()</code> | 每次构造独立 DynamicCache，避免两种来源世界互相污染。 |
| 42 | <code>    with attention_backend(model, "sdpa"):</code> | prefill 统一使用 SDPA，并按上下文管理器恢复后端。 |
| 43 | <code>        for start in range(0, len(tokens), PREFILL_CHUNK):</code> | 按 256 步长遍历前缀，有限次结束；空 tokens 则返回空 cache。 |
| 44 | <code>            model.model(</code> | 调用主干 model.model，不经过 lm_head。 |
| 45 | <code>                input_ids=input_ids(model, tokens[start:start + PREFILL_CHUNK]),</code> | 当前输入 [1,C]，C≤256；与历史 cache 一起构成完整因果前缀。 |
| 46 | <code>                past_key_values=cache, use_cache=True, return_dict=True,</code> | 传入同一 cache 并 use_cache=True，使其原地增长；丢弃输出隐藏态，保留缓存副作用。 |
| 47 | <code>            )</code> | 传入同一 cache 并 use_cache=True，使其原地增长；丢弃输出隐藏态，保留缓存副作用。 |
| 48 | <code>    return cache</code> | cache 所有权交给调用者，后续 scoring 会继续修改它。 |
| 49 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 50 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 51 | <code>def score_continuation(model, cache, last_prompt_token, answer, on_logits=None):</code> | score_continuation 输入已有前缀 cache、最后提示 token、回答及可选 logits 回调；返回 [N] CPU logp，原地追加 KV。 |
| 52 | <code>    """Query P+t-1 predicts y_t; local callers reset the prefix before a unit."""</code> | score_continuation 输入已有前缀 cache、最后提示 token、回答及可选 logits 回调；返回 [N] CPU logp，原地追加 KV。 |
| 53 | <code>    queries = [last_prompt_token, *answer[:-1]]</code> | 查询是最后提示 token 加 y 的前 N−1 项，所以位置 P+t−1 预测 y_t，而不是用 y_t 自己预测自己。 |
| 54 | <code>    likelihoods = []</code> | 分块收集 CPU logp；空回答会在 concatenate 失败，公共文本入口先拒绝空 token 回答。 |
| 55 | <code>    with attention_backend(model, "sdpa"):</code> | 似然查询使用 SDPA，但不保证某个具体融合 kernel 已执行。 |
| 56 | <code>        for start in range(0, len(answer), QUERY_CHUNK):</code> | 按最多 16 个回答位置循环，保留固定 teacher-forced 历史。 |
| 57 | <code>            hidden = model.model(</code> | 主干继续追加 cache 并返回本块隐藏态。 |
| 58 | <code>                input_ids=input_ids(model, queries[start:start + QUERY_CHUNK]),</code> | 送入 [1,C] 的预测查询 token，不送对应目标 token 的最后一项。 |
| 59 | <code>                past_key_values=cache, use_cache=True, return_dict=True,</code> | 复用并增长调用者 cache，return_dict 保证命名隐藏态接口。 |
| 60 | <code>            ).last_hidden_state[0]</code> | 取 batch 0 后 hidden 为 [C,D]；C 是本块位置数，D 是隐藏维度。 |
| 61 | <code>            native = model.lm_head(hidden)</code> | 原生 BF16 lm_head 得到 [C,V] logits；先计算投影再转 FP32，不能交换顺序冒充相同数值。 |
| 62 | <code>            if on_logits is not None:</code> | 若需要全词表诊断，将原生 logits 和本块起点交给回调；回调应自行 detach/保存。 |
| 63 | <code>                on_logits(start, native)</code> | 若需要全词表诊断，将原生 logits 和本块起点交给回调；回调应自行 detach/保存。 |
| 64 | <code>            logits = native.float()</code> | 只把已计算 logits 转 FP32，用于稳定归一化和目标似然。 |
| 65 | <code>            targets = input_ids(model, answer[start:start + QUERY_CHUNK])[0]</code> | 目标为同一回答位置的 [C] int64 token IDs。 |
| 66 | <code>            actual = logits.gather(1, targets[:, None])[:, 0]</code> | 按词表轴 gather 每行实际 token 的 logit，输出 [C]。 |
| 67 | <code>            likelihoods.append((actual - logits.logsumexp(-1)).cpu().numpy())</code> | 减每行 logsumexp 得 logp，移到 CPU 转 NumPy；要求外层无梯度，否则 numpy 可能拒绝。 |
| 68 | <code>    return np.concatenate(likelihoods)</code> | 按原始回答位置拼接所有块，不改变 token 顺序。 |
