# myproject/counterfactual.py 逐行讲解

基线：`1210120748b3a05bd271bc79debcbed4fdc0c964`；SHA-256：`2c1e0cc6f800a1308fd42689b3739dc03592902413f713fd36fc9ef8aa6430da`；54 行。

源码与注释逐行并排。一个跨行表达式共用同一操作说明；这不代表各行独立执行。完整调用路径、输入输出轴和理论边界见 [METHOD.md](../METHOD.md)。未执行的 GPU 路径见 [验证边界](../VALIDATION.md)。

| 函数（含嵌套） | 原始行号 | 作用与契约入口 |
|---|---|---|
| `observe_responses` | 10–30 | observe_responses 接收来源编码和非空回答ID列表，返回每回答 [N_j,2] logp；两列 present/absent，同为SDPA，且每变体有自己的原token轴。由 API.measure_responses 调用。 |
| `repair_support` | 33–54 | repair_support 输入原回答未来 [F,2] 与候选 [C,F,2]；返回来源×修复交互、增量、候选增益/absent权重。只讨论预测支持，不能解释成修复正确性。 |

| 行 | 原始代码 | 操作、目的与边界 |
|---|---|---|
| 1 | <code>"""Matched response interventions and retrospective source-support interactions."""</code> | 匹配后端的回答变体观测与回顾式来源交互；不是默认检测器读出。 |
| 2 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 3 | <code>import numpy as np</code> | NumPy 处理逐变体 logp 轴和累计和。 |
| 4 | <code>from scipy.special import logsumexp</code> | logsumexp 在序列级混合候选，避免把概率/密度任意逐token平均。 |
| 5 | <code>import torch</code> | torch.no_grad 保护变体观测。 |
| 6 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 7 | <code>from .model import prefill, score_continuation</code> | 复用相同 SDPA scorer/prefill，不经过带路由的 eager 观测器。 |
| 8 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 9 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 10 | <code>@torch.no_grad()</code> | observe_responses 接收来源编码和非空回答ID列表，返回每回答 [N_j,2] logp；两列 present/absent，同为SDPA，且每变体有自己的原token轴。由 API.measure_responses 调用。 |
| 11 | <code>def observe_responses(model, source, answers):</code> | observe_responses 接收来源编码和非空回答ID列表，返回每回答 [N_j,2] logp；两列 present/absent，同为SDPA，且每变体有自己的原token轴。由 API.measure_responses 调用。 |
| 12 | <code>    """Teacher-force supplied responses with two prompt prefills, both SDPA.</code> | observe_responses 接收来源编码和非空回答ID列表，返回每回答 [N_j,2] logp；两列 present/absent，同为SDPA，且每变体有自己的原token轴。由 API.measure_responses 调用。 |
| 13 | <code> </code> | observe_responses 接收来源编码和非空回答ID列表，返回每回答 [N_j,2] logp；两列 present/absent，同为SDPA，且每变体有自己的原token轴。由 API.measure_responses 调用。 |
| 14 | <code>    Reuse only the original prompt KV prefix. Columns are source-present and</code> | observe_responses 接收来源编码和非空回答ID列表，返回每回答 [N_j,2] logp；两列 present/absent，同为SDPA，且每变体有自己的原token轴。由 API.measure_responses 调用。 |
| 15 | <code>    source-absent log probabilities on each response's own raw token axis.</code> | observe_responses 接收来源编码和非空回答ID列表，返回每回答 [N_j,2] logp；两列 present/absent，同为SDPA，且每变体有自己的原token轴。由 API.measure_responses 调用。 |
| 16 | <code>    No routing observer, labels, generation or factual judge is involved.</code> | observe_responses 接收来源编码和非空回答ID列表，返回每回答 [N_j,2] logp；两列 present/absent，同为SDPA，且每变体有自己的原token轴。由 API.measure_responses 调用。 |
| 17 | <code>    """</code> | observe_responses 接收来源编码和非空回答ID列表，返回每回答 [N_j,2] logp；两列 present/absent，同为SDPA，且每变体有自己的原token轴。由 API.measure_responses 调用。 |
| 18 | <code>    if not answers or any(not len(answer) for answer in answers):</code> | 要求至少一个回答且每回答非空，防止底层 concatenate 空数组。 |
| 19 | <code>        raise ValueError('Provide nonempty response token sequences')</code> | 要求至少一个回答且每回答非空，防止底层 concatenate 空数组。 |
| 20 | <code>    worlds = []</code> | 按来源世界保存每个变体的结果列表。 |
| 21 | <code>    for name in ['prompt_with_source', 'prompt_without_source']:</code> | 顺序测 present/absent，各自只 prefill 一次提示P−1前缀，绝不跨世界共用cache。 |
| 22 | <code>        prompt = source[name]</code> | 顺序测 present/absent，各自只 prefill 一次提示P−1前缀，绝不跨世界共用cache。 |
| 23 | <code>        cache = prefill(model, prompt[:-1])</code> | 顺序测 present/absent，各自只 prefill 一次提示P−1前缀，绝不跨世界共用cache。 |
| 24 | <code>        rows = []</code> | 每世界遍历同一组已给定回答，不采样生成候选。 |
| 25 | <code>        for answer in answers:</code> | 每世界遍历同一组已给定回答，不采样生成候选。 |
| 26 | <code>            cache.crop(len(prompt)-1)</code> | 每个变体前裁回提示P−1，避免上个回答历史泄漏。 |
| 27 | <code>            rows.append(score_continuation(model, cache, prompt[-1], answer))</code> | 用固定右移 teacher forcing 读取该变体自己的完整回答 logp[N_j]。 |
| 28 | <code>        worlds.append(rows)</code> | 保存该来源世界的所有变体结果。 |
| 29 | <code>    return [np.column_stack(pair).astype(np.float64)</code> | 按变体严格配对两世界，column_stack 得 [N_j,2] 后转float64；原始 logp 仍来自 FP32 运算。 |
| 30 | <code>            for pair in zip(*worlds, strict=True)]</code> | 按变体严格配对两世界，column_stack 得 [N_j,2] 后转float64；原始 logp 仍来自 FP32 运算。 |
| 31 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 32 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 33 | <code>def repair_support(original, alternatives):</code> | repair_support 输入原回答未来 [F,2] 与候选 [C,F,2]；返回来源×修复交互、增量、候选增益/absent权重。只讨论预测支持，不能解释成修复正确性。 |
| 34 | <code>    """Source-by-response-repair interaction on an unchanged future sequence.</code> | repair_support 输入原回答未来 [F,2] 与候选 [C,F,2]；返回来源×修复交互、增量、候选增益/absent权重。只讨论预测支持，不能解释成修复正确性。 |
| 35 | <code> </code> | repair_support 输入原回答未来 [F,2] 与候选 [C,F,2]；返回来源×修复交互、增量、候选增益/absent权重。只讨论预测支持，不能解释成修复正确性。 |
| 36 | <code>    Inputs: [future, present/absent] and [candidate, future, present/absent].</code> | repair_support 输入原回答未来 [F,2] 与候选 [C,F,2]；返回来源×修复交互、增量、候选增益/absent权重。只讨论预测支持，不能解释成修复正确性。 |
| 37 | <code>    Uniform candidate priors are mixed at SEQUENCE level in each world.</code> | repair_support 输入原回答未来 [F,2] 与候选 [C,F,2]；返回来源×修复交互、增量、候选增益/absent权重。只讨论预测支持，不能解释成修复正确性。 |
| 38 | <code>    R = (log mixture_present - log mixture_absent) - original source support.</code> | repair_support 输入原回答未来 [F,2] 与候选 [C,F,2]；返回来源×修复交互、增量、候选增益/absent权重。只讨论预测支持，不能解释成修复正确性。 |
| 39 | <code>    This signed predictive interaction is not a factual-error probability.</code> | repair_support 输入原回答未来 [F,2] 与候选 [C,F,2]；返回来源×修复交互、增量、候选增益/absent权重。只讨论预测支持，不能解释成修复正确性。 |
| 40 | <code>    """</code> | repair_support 输入原回答未来 [F,2] 与候选 [C,F,2]；返回来源×修复交互、增量、候选增益/absent权重。只讨论预测支持，不能解释成修复正确性。 |
| 41 | <code>    original, alternatives = np.asarray(original, float), np.asarray(alternatives, float)</code> | 两个输入转浮点，以一致轴处理序列级概率。 |
| 42 | <code>    if (original.ndim != 2 or original.shape[1] != 2 or not len(original)</code> | 要求非空未来、非空候选、相同未来与世界轴且全部有限；不支持未对齐的不同长度未来。 |
| 43 | <code>            or alternatives.ndim != 3 or not len(alternatives)</code> | 要求非空未来、非空候选、相同未来与世界轴且全部有限；不支持未对齐的不同长度未来。 |
| 44 | <code>            or alternatives.shape[1:] != original.shape</code> | 要求非空未来、非空候选、相同未来与世界轴且全部有限；不支持未对齐的不同长度未来。 |
| 45 | <code>            or not np.isfinite(original).all() or not np.isfinite(alternatives).all()):</code> | 要求非空未来、非空候选、相同未来与世界轴且全部有限；不支持未对齐的不同长度未来。 |
| 46 | <code>        raise ValueError('Matching finite nonempty future/candidate likelihood axes required')</code> | 要求非空未来、非空候选、相同未来与世界轴且全部有限；不支持未对齐的不同长度未来。 |
| 47 | <code>    cumulative = np.cumsum(alternatives, axis=1)</code> | 沿未来位置累计每候选 logp，得到各前缀序列 log likelihood。 |
| 48 | <code>    mixture = logsumexp(cumulative, axis=0)-np.log(len(alternatives))</code> | 在候选轴 logsumexp−log C，使用均匀先验混合完整前缀序列；不是每步重新平均候选。 |
| 49 | <code>    baseline = np.cumsum(original[:, 0]-original[:, 1])</code> | 原回答来源支持是累计 present−absent，符号与检测 d 相反。 |
| 50 | <code>    contrast = mixture[:, 0]-mixture[:, 1]-baseline</code> | 候选混合的来源支持减原回答来源支持，得到逐未来前缀交互。 |
| 51 | <code>    totals = alternatives.sum(axis=1)</code> | 各候选整段 logp 在未来轴求和。 |
| 52 | <code>    return {'score': float(contrast[-1]), 'increments': np.diff(np.r_[0., contrast]),</code> | 最终交互取最后前缀；increments 对前补0后的交互差分，可恢复完整总和。 |
| 53 | <code>            'candidate_gains': totals[:, 0]-totals[:, 1]-baseline[-1],</code> | 每候选整段来源支持减原baseline，保留个体而不只返回混合。 |
| 54 | <code>            'absent_weights': np.exp(totals[:, 1]-logsumexp(totals[:, 1]))}</code> | 按 absent 世界序列likelihood归一候选权重；它描述混合机制，不是候选事实正确概率。 |
