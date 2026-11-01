# myproject/inputs.py 逐行讲解

基线：`1210120748b3a05bd271bc79debcbed4fdc0c964`；SHA-256：`e0d7c60577905214ade564fa17faf9c724cbbb01ad90e89837e727886f4e68a5`；71 行。

源码与注释逐行并排。一个跨行表达式共用同一操作说明；这不代表各行独立执行。完整调用路径、输入输出轴和理论边界见 [METHOD.md](../METHOD.md)。未执行的 GPU 路径见 [验证边界](../VALIDATION.md)。

| 函数（含嵌套） | 原始行号 | 作用与契约入口 |
|---|---|---|
| `text_units` | 12–33 | text_units 接收逐 token decode 片段，按句界并限长切出原 token 半开范围，返回 units。max_tokens 默认 64；非正参数没有独立验证。 |
| `encode_source` | 36–59 | encode_source 输入 tokenizer、完整用户提示和来源子串；输出 present IDs、来源 mask、absent IDs。由 API.measure/measure_responses 调用。 |
| `encode_response` | 62–71 | encode_response 输入已有文本，输出原 IDs、offsets、原文、特殊 ID 集合及默认 units；不生成或纠正回答。 |

| 行 | 原始代码 | 操作、目的与边界 |
|---|---|---|
| 1 | <code>"""Encode a prompt, its source passage and an existing response."""</code> | 编码已有 prompt/来源/回答，建立唯一原始 token/字符坐标。 |
| 2 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 3 | <code>import re</code> | 正则用于文本标点和裸列表标记。 |
| 4 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 5 | <code>import numpy as np</code> | NumPy 提供字符累计长度、二分查找和 offset 相交判断。 |
| 6 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 7 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 8 | <code>BOUNDARY = re.compile(r'\n+&#124;[.!?。！？]+["”’\')\]]*(?=\s&#124;$)')</code> | 匹配换行，或中英文句末标点及闭合引号/括号；标点后要求空白或末尾，故不是通用中文句法切分器。 |
| 9 | <code>LIST_MARKER = re.compile(r'\s*(?:\d+[.)]&#124;[-*•])\s*')</code> | 匹配仅由数字编号或项目符号组成的片段，防止把 '1.' 当完整句。 |
| 10 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 11 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 12 | <code>def text_units(token_text, max_tokens=64):</code> | text_units 接收逐 token decode 片段，按句界并限长切出原 token 半开范围，返回 units。max_tokens 默认 64；非正参数没有独立验证。 |
| 13 | <code>    """Group at original newlines/sentence ends; cap local contexts at 64 tokens."""</code> | text_units 接收逐 token decode 片段，按句界并限长切出原 token 半开范围，返回 units。max_tokens 默认 64；非正参数没有独立验证。 |
| 14 | <code>    text = ''.join(token_text)</code> | 拼接逐 token decode 文本仅用于分段；它不保证等于原 response，因此最终显示仍用原 offsets。 |
| 15 | <code>    ends = np.cumsum([len(piece) for piece in token_text])</code> | 累加每个解码片段字符长度，ends[k] 表示该 token 在拼接文本中的结束位置。 |
| 16 | <code>    boundaries = [0]</code> | 切点从 token 0 开始。 |
| 17 | <code>    text_start = 0</code> | text_start 是当前候选句子的字符起点，不是 token 下标。 |
| 18 | <code>    for match in BOUNDARY.finditer(text):</code> | 从拼接文本中顺序找句界，不使用事实标签。 |
| 19 | <code>        fragment = text[text_start:match.end()]</code> | 取上次边界后的候选文本，包含本次匹配的标点/换行。 |
| 20 | <code>        if fragment.strip() and not LIST_MARKER.fullmatch(fragment):</code> | 排除纯空白和裸列表编号，其他候选才形成边界。 |
| 21 | <code>            stop = int(np.searchsorted(ends, match.end(), side='left')) + 1</code> | 将字符结束点映射到覆盖它的 token 后一个位置；不在 token 内强切。 |
| 22 | <code>            if stop &gt; boundaries[-1]:</code> | 只有边界向前移动才追加，同时把字符起点移到完整 token 末尾，避免重复/退回。 |
| 23 | <code>                boundaries.append(stop)</code> | 只有边界向前移动才追加，同时把字符起点移到完整 token 末尾，避免重复/退回。 |
| 24 | <code>                text_start = int(ends[stop - 1])</code> | 只有边界向前移动才追加，同时把字符起点移到完整 token 末尾，避免重复/退回。 |
| 25 | <code>    if boundaries[-1] != len(token_text):</code> | 最后一句即使没有句末标点也要覆盖到 token 总数。 |
| 26 | <code>        boundaries.append(len(token_text))</code> | 最后一句即使没有句末标点也要覆盖到 token 总数。 |
| 27 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 28 | <code>    units = []</code> | 准备最终局部片段列表。 |
| 29 | <code>    for left, right in zip(boundaries, boundaries[1:]):</code> | 遍历自然文本单元的 token 范围。 |
| 30 | <code>        for start in range(left, right, max_tokens):</code> | 再按 max_tokens 步长将长单元切为多个局部片段；max_tokens=0 会由 range 报错。 |
| 31 | <code>            units.append(dict(start=start, stop=min(start + max_tokens, right),</code> | 每局部片段 start/stop 是原 token 下标，stop 不越过自然单元右边界。 |
| 32 | <code>                              text_start=left, text_stop=right))</code> | text_start/text_stop 实际存自然单元的 token 范围，名称易误以为字符 offset；应按代码而非名称解释。 |
| 33 | <code>    return units</code> | 返回覆盖原 token 的有序 units；空 token 输入返回空列表。 |
| 34 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 35 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 36 | <code>def encode_source(tokenizer, prompt: str, source: str) -&gt; dict:</code> | encode_source 输入 tokenizer、完整用户提示和来源子串；输出 present IDs、来源 mask、absent IDs。由 API.measure/measure_responses 调用。 |
| 37 | <code>    """Mark the unique source passage inside the complete user prompt."""</code> | encode_source 输入 tokenizer、完整用户提示和来源子串；输出 present IDs、来源 mask、absent IDs。由 API.measure/measure_responses 调用。 |
| 38 | <code>    if not source or prompt.count(source) != 1:</code> | 来源必须非空且在 prompt 中唯一出现，避免无法判定删除哪一段。 |
| 39 | <code>        raise ValueError("source must be a nonempty substring occurring once in prompt")</code> | 来源必须非空且在 prompt 中唯一出现，避免无法判定删除哪一段。 |
| 40 | <code>    chat = tokenizer.apply_chat_template(</code> | 先渲染单 user 消息的 chat template，并加 assistant 生成提示，不先让模板分词。 |
| 41 | <code>        [dict(role="user", content=prompt)], tokenize=False, add_generation_prompt=True,</code> | 先渲染单 user 消息的 chat template，并加 assistant 生成提示，不先让模板分词。 |
| 42 | <code>    )</code> | 先渲染单 user 消息的 chat template，并加 assistant 生成提示，不先让模板分词。 |
| 43 | <code>    if prompt not in chat:</code> | 原 prompt 若未原样保留，尝试处理模板对消息边缘空白的 trim。 |
| 44 | <code>        # Llama templates trim message edges. Align to the content actually sent</code> | 注释限定这个修正只为对齐实际发送内容，同时完整来源必须保留。 |
| 45 | <code>        # to the model, while requiring the complete evidence to remain intact.</code> | 注释限定这个修正只为对齐实际发送内容，同时完整来源必须保留。 |
| 46 | <code>        prompt = prompt.strip()</code> | 只去掉 prompt 首尾空白，不改中间来源。 |
| 47 | <code>        if prompt.count(source) != 1:</code> | 去边缘后若来源不再唯一则拒绝；若模板作其他实质改写，后续 chat.index 仍可能报错。 |
| 48 | <code>            raise ValueError('Chat template altered the source passage')</code> | 去边缘后若来源不再唯一则拒绝；若模板作其他实质改写，后续 chat.index 仍可能报错。 |
| 49 | <code>    start = prompt.index(source)</code> | 找到来源在实际用户内容中的字符起点。 |
| 50 | <code>    stop = start + len(source)</code> | 来源字符 stop 是半开区间结束。 |
| 51 | <code>    shift = chat.index(prompt)</code> | 找到用户内容在整个 chat 模板中的字符偏移；模板内重复内容的极端情况此处取第一次。 |
| 52 | <code>    encoded = tokenizer(chat, add_special_tokens=False, return_offsets_mapping=True)</code> | 对已渲染 chat 分词且不再加特殊符号，取得每个 token 的字符 offset。 |
| 53 | <code>    offsets = np.asarray(encoded["offset_mapping"])</code> | 将 offsets 变为 [P,2] 数组。 |
| 54 | <code>    mask = (offsets[:, 0] &lt; shift + stop) &amp; (offsets[:, 1] &gt; shift + start)</code> | 来源 mask 选择与来源字符区间有正交集的整枚 token，边界 token 可能还含邻接非来源字符。 |
| 55 | <code>    return {</code> | 保留完整 present token IDs 与同长 bool 来源 mask。 |
| 56 | <code>        "prompt_with_source": encoded["input_ids"], "source_mask": mask.tolist(),</code> | 保留完整 present token IDs 与同长 bool 来源 mask。 |
| 57 | <code>        "prompt_without_source": [token for token, remove in zip(encoded["input_ids"], mask)</code> | 删除 mask 命中的整枚 token，保留其他 token 原 ID；不把剩余字符串重新分词。 |
| 58 | <code>                                  if not remove],</code> | 删除 mask 命中的整枚 token，保留其他 token 原 ID；不把剩余字符串重新分词。 |
| 59 | <code>    }</code> | 返回编码字典；该删除操作本身不是合法条件随机化抽样。 |
| 60 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 61 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 62 | <code>def encode_response(tokenizer, response: str) -&gt; dict:</code> | encode_response 输入已有文本，输出原 IDs、offsets、原文、特殊 ID 集合及默认 units；不生成或纠正回答。 |
| 63 | <code>    """Retain original response tokens, character offsets and local text units."""</code> | encode_response 输入已有文本，输出原 IDs、offsets、原文、特殊 ID 集合及默认 units；不生成或纠正回答。 |
| 64 | <code>    encoded = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)</code> | 回答分词时禁用额外特殊 token，offset_mapping 绑定原文字符。 |
| 65 | <code>    token_text = [tokenizer.decode([token]) for token in encoded["input_ids"]]</code> | 逐枚 decode 供句界识别，可能有 Unicode 字节片段差异；不用于最终字符输出。 |
| 66 | <code>    marked = {index for index, token in tokenizer.added_tokens_decoder.items() if token.special}</code> | 从 added token 元信息额外收集 special=True 的 ID。 |
| 67 | <code>    special_ids = sorted(set(tokenizer.all_special_ids) &#124; marked)</code> | 与 tokenizer.all_special_ids 合并去重排序，后续只过滤这些特殊 token 及空 offset。 |
| 68 | <code>    return {</code> | 保存原始 token IDs 和每枚 token 的 [start,end) 字符范围。 |
| 69 | <code>        "answer_ids": encoded["input_ids"], "offsets": encoded["offset_mapping"],</code> | 保存原始 token IDs 和每枚 token 的 [start,end) 字符范围。 |
| 70 | <code>        "text": response, "special_ids": special_ids, "units": text_units(token_text),</code> | 保存原文和特殊 ID，并使用默认 64-token 上限构造标点片段。 |
| 71 | <code>    }</code> | 返回可供所有来源条件共同复用的编码对象。 |
