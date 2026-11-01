# myproject/proofs.py 逐行讲解

基线：`1210120748b3a05bd271bc79debcbed4fdc0c964`；SHA-256：`191727f0f2e073b3ab53ffea7ba4d2e946514a771f6afd4d9d026c70d63de7b9`；554 行。

源码与注释逐行并排。一个跨行表达式共用同一操作说明；这不代表各行独立执行。完整调用路径、输入输出轴和理论边界见 [METHOD.md](../METHOD.md)。未执行的 GPU 路径见 [验证边界](../VALIDATION.md)。

| 函数（含嵌套） | 原始行号 | 作用与契约入口 |
|---|---|---|
| `_name` | 26–27 | _name把对象转字符串、NFKC、casefold并压缩空白，供表行名/单位匹配；不同表面名可能归一相同，不声称实体消歧。 |
| `_decimal` | 30–39 | _decimal接收数值字面字符串，返回Decimal或解析失败None；供_amount使用。 |
| `_unit` | 42–55 | _unit要求整段都是可识别单位，返回(currency元组,scale元组,percent布尔)；空签名表示未说明，None表示无效/冲突。 |
| `_merge_unit` | 58–69 | _merge_unit合并表头/列头/金额上的兼容显式单位，返回签名或None；调用处必须检查未知。 |
| `_has_unit` | 72–73 | _has_unit要求签名有效且至少一部分明确；全空单位不够支撑证书。 |
| `_amount_pattern` | 76–79 | _amount_pattern(prefix)生成具命名分组的金额正则，prefix避免同一二元表达式的多金额分组重名。 |
| `_amount` | 82–93 | _amount将regex match的指定金额组转Decimal、单位和字符role范围；由来源单元格和回答断言共用。 |
| `_header_unit` | 96–104 | _header_unit只接受完整单位或明确的末尾单位括号，不从任意标题猜单位。 |
| `_json_tables` | 107–151 | _json_tables是生成器，扫描来源中以字面'[['开头的JSON表，并返回表起止、行值和每cell原文span；不支持所有可能JSON排版。 |
| `_source_scopes` | 154–226 | _source_scopes将来源表编译为封闭年份/行/单位事实域，返回scopes与结构问题；由compile_proofs调用，不读取标签。 |
| `_anchors` | 229–237 | _anchors在回答中按有限来源subjects找字面锚点，返回字符起止及subject；不做开放实体抽取。 |
| `_claims` | 240–266 | _claims从字面锚点编译允许的直接/二元断言，返回regex matches与schema/spans；未识别句子不会强行解析。 |
| `_assertion_scope` | 269–295 | _assertion_scope要求整个回答由有限声明句组成；任意前言/跟进可能引入否定、引用、假设，因此不能只抽到一句就当真实断言。 |
| `_resolved_unit` | 298–300 | _resolved_unit在scope的已知单位中找与observed可合并成同单位的候选，只有唯一候选才返回；缺失/歧义返回None，不任选。 |
| `_terms` | 303–309 | _terms规范化单调DNF：输入role序号项，返回无重复、无被包含冗余项的最小修复集合列表。 |
| `_terms.<lambda>@304:73` | 304–304 | 每term内部去重排序，全体去重；嵌套lambda按(term长度,字典序)排序，小集合先处理。 |
| `_role_tokens` | 312–336 | _role_tokens把每role字符spans映射到selected有效token轴，返回每role局部索引列表或None；不能将跨role token硬拆。 |
| `_numeric_roles` | 339–343 | _numeric_roles建立数值幅度role，只有出现显式符号字符时另建_sign role；没有符号不能造一个不存在的token作为修复对象。 |
| `_numeric_repairs` | 346–353 | _numeric_repairs对观察金额与候选见证Decimal比较，返回须修改的role序号及无法投影的隐式符号变化标志。 |
| `_arithmetic` | 356–360 | _arithmetic在局部Decimal context按最大整数位和最小小数指数设置足够精度，防止默认精度舍入金额运算。 |
| `_plausibly_rounded` | 363–369 | _plausibly_rounded判断表中精确值与显示更低精度金额的差异是否可能只是量化，用来弃权而非判错误。 |
| `_evaluate` | 372–490 | _evaluate是声明×封闭来源scope的核心：返回roles、最小repair terms、见证trace、状态与原因；不修改分数。 |
| `compile_proofs` | 493–554 | compile_proofs是外部入口，输入source/response、原Nx2 offsets、有效target mask或不同整型索引及可选question；返回不兼容DNF组件与中立覆盖诊断。未解析/歧义/支持断言不标任何token为clean。由revision.proof_pack调用。 |

| 行 | 原始代码 | 操作、目的与边界 |
|---|---|---|
| 1 | <code>"""Conservative certificates for a finite, explicit table/answer grammar.</code> | 有限语法证书编译器：只理解嵌入JSON行表、字面年份/单位和规定的直接/二元算术断言；未解析散文中立。返回token索引指向offsets[target]的有效轴，不是原始全token轴。 |
| 2 | <code> </code> | 有限语法证书编译器：只理解嵌入JSON行表、字面年份/单位和规定的直接/二元算术断言；未解析散文中立。返回token索引指向offsets[target]的有效轴，不是原始全token轴。 |
| 3 | <code>This is not a semantic parser. Only embedded JSON row tables with literal year</code> | 有限语法证书编译器：只理解嵌入JSON行表、字面年份/单位和规定的直接/二元算术断言；未解析散文中立。返回token索引指向offsets[target]的有效轴，不是原始全token轴。 |
| 4 | <code>columns and explicit units are understood. Direct statements use ``ROW in YEAR</code> | 有限语法证书编译器：只理解嵌入JSON行表、字面年份/单位和规定的直接/二元算术断言；未解析散文中立。返回token索引指向offsets[target]的有效轴，不是原始全token轴。 |
| 5 | <code>was AMOUNT`` (also year-first and amount-before-year). Binary statements use</code> | 有限语法证书编译器：只理解嵌入JSON行表、字面年份/单位和规定的直接/二元算术断言；未解析散文中立。返回token索引指向offsets[target]的有效轴，不是原始全token轴。 |
| 6 | <code>``ROW: AMOUNT in YEAR +/- AMOUNT in YEAR = AMOUNT``. Unparsed prose is neutral.</code> | 有限语法证书编译器：只理解嵌入JSON行表、字面年份/单位和规定的直接/二元算术断言；未解析散文中立。返回token索引指向offsets[target]的有效轴，不是原始全token轴。 |
| 7 | <code>All returned token indices address ``offsets[target]``, in the supplied order.</code> | 有限语法证书编译器：只理解嵌入JSON行表、字面年份/单位和规定的直接/二元算术断言；未解析散文中立。返回token索引指向offsets[target]的有效轴，不是原始全token轴。 |
| 8 | <code>"""</code> | 有限语法证书编译器：只理解嵌入JSON行表、字面年份/单位和规定的直接/二元算术断言；未解析散文中立。返回token索引指向offsets[target]的有效轴，不是原始全token轴。 |
| 9 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 10 | <code>from collections import Counter</code> | Counter统计supported/incompatible/unknown及弃权原因。 |
| 11 | <code>from decimal import Decimal, InvalidOperation, localcontext</code> | Decimal避免金额二进制浮点误差；InvalidOperation用于解析失败；localcontext按实际位数设算术精度。 |
| 12 | <code>import json</code> | 标准JSONDecoder读取表及每个单元格的原文范围。 |
| 13 | <code>import re</code> | 正则仅实现明确有限语法，不作开放语义解析。 |
| 14 | <code>import unicodedata</code> | unicodedata的NFKC规范化名字/单位表面形式。 |
| 15 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 16 | <code>import numpy as np</code> | NumPy管理原token offsets/target与范围验证。 |
| 17 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 18 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 19 | <code>_YEAR = r"(?:19&#124;20)\d{2}"</code> | 年份只承认1900–2099的四位字面量，不推断'去年'或其他纪年。 |
| 20 | <code>_NUM = r"(?:\([+-]?\d+(?:,\d{3})*(?:\.\d+)?\)&#124;[+-]?\d+(?:,\d{3})*(?:\.\d+)?)"</code> | 金额字面语法允许显式正负号、会计括号、逗号和小数，不支持科学计数法或自然语言数词。 |
| 21 | <code>_SCALE = r"(?:thousand&#124;million&#124;billion)s?"</code> | 尺度只支持thousand/million/billion及复数，不自动推断隐含单位。 |
| 22 | <code>_MAX_PROGRAMS = 4096</code> | 二元见证程序最多4096、最小修复DNF项最多256；超出返回unknown，不截断后假称穷尽。 |
| 23 | <code>_MAX_TERMS = 256</code> | 二元见证程序最多4096、最小修复DNF项最多256；超出返回unknown，不截断后假称穷尽。 |
| 24 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 25 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 26 | <code>def _name(text):</code> | _name把对象转字符串、NFKC、casefold并压缩空白，供表行名/单位匹配；不同表面名可能归一相同，不声称实体消歧。 |
| 27 | <code>    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())</code> | _name把对象转字符串、NFKC、casefold并压缩空白，供表行名/单位匹配；不同表面名可能归一相同，不声称实体消歧。 |
| 28 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 29 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 30 | <code>def _decimal(text):</code> | _decimal接收数值字面字符串，返回Decimal或解析失败None；供_amount使用。 |
| 31 | <code>    text = text.strip().replace(",", "")</code> | 去边缘空白和分组逗号。 |
| 32 | <code>    if text.startswith("(") and text.endswith(")"):</code> | 外围括号按会计负数处理；若内文已有显式正负号则不再加负号。二元括号歧义另在_evaluate弃权。 |
| 33 | <code>        text = text[1:-1]</code> | 外围括号按会计负数处理；若内文已有显式正负号则不再加负号。二元括号歧义另在_evaluate弃权。 |
| 34 | <code>        if not text.startswith(("-", "+")):</code> | 外围括号按会计负数处理；若内文已有显式正负号则不再加负号。二元括号歧义另在_evaluate弃权。 |
| 35 | <code>            text = "-" + text</code> | 外围括号按会计负数处理；若内文已有显式正负号则不再加负号。二元括号歧义另在_evaluate弃权。 |
| 36 | <code>    try:</code> | 尝试Decimal解析，InvalidOperation转None，不吞其他编程异常。 |
| 37 | <code>        return Decimal(text)</code> | 尝试Decimal解析，InvalidOperation转None，不吞其他编程异常。 |
| 38 | <code>    except InvalidOperation:</code> | 尝试Decimal解析，InvalidOperation转None，不吞其他编程异常。 |
| 39 | <code>        return None</code> | 尝试Decimal解析，InvalidOperation转None，不吞其他编程异常。 |
| 40 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 41 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 42 | <code>def _unit(text):</code> | _unit要求整段都是可识别单位，返回(currency元组,scale元组,percent布尔)；空签名表示未说明，None表示无效/冲突。 |
| 43 | <code>    """Parse an entire unit/header literal; an empty signature is unspecified."""</code> | _unit要求整段都是可识别单位，返回(currency元组,scale元组,percent布尔)；空签名表示未说明，None表示无效/冲突。 |
| 44 | <code>    text = _name(text)</code> | 先统一大小写、兼容Unicode和空白。 |
| 45 | <code>    currency = re.findall(r"[$£€]&#124;\b(?:usd&#124;gbp&#124;eur)\b", text)</code> | 只抽美元/英镑/欧元符号及usd/gbp/eur字面；'$'与'USD'没有自动归成同一个规范货币。 |
| 46 | <code>    scales = re.findall(rf"\b{_SCALE}\b", text)</code> | 抽可支持的尺度单词。 |
| 47 | <code>    percent = re.findall(r"%&#124;\bpercent\b", text)</code> | 抽百分号或percent。 |
| 48 | <code>    signature = (tuple(sorted(set(currency))),</code> | 货币/尺度去重排序形成hashable签名，尺度去复数s；是否百分比为第三项。 |
| 49 | <code>                 tuple(sorted({s.rstrip("s") for s in scales})), bool(percent))</code> | 货币/尺度去重排序形成hashable签名，尺度去复数s；是否百分比为第三项。 |
| 50 | <code>    stripped = re.sub(rf"[$£€%()]&#124;\b(?:usd&#124;gbp&#124;eur&#124;percent&#124;in&#124;{_SCALE})\b", "", text)</code> | 移除允许的单位符号、括号、in与尺度词，剩余文本用于整段识别门槛。 |
| 51 | <code>    if stripped.strip() or len(signature[0]) &gt; 1 or len(signature[1]) &gt; 1:</code> | 还有非空未知文字，或多个不同货币/尺度，则拒绝这个单位。 |
| 52 | <code>        return None</code> | 还有非空未知文字，或多个不同货币/尺度，则拒绝这个单位。 |
| 53 | <code>    if signature[2] and (signature[0] or signature[1]):</code> | 百分比不能同时带货币或数量尺度，冲突即None。 |
| 54 | <code>        return None</code> | 百分比不能同时带货币或数量尺度，冲突即None。 |
| 55 | <code>    return signature</code> | 返回签名，不换算不同尺度间的数值。 |
| 56 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 57 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 58 | <code>def _merge_unit(left, right):</code> | _merge_unit合并表头/列头/金额上的兼容显式单位，返回签名或None；调用处必须检查未知。 |
| 59 | <code>    if left is None or right is None:</code> | 任一输入已无效则不能恢复，返回None。 |
| 60 | <code>        return None</code> | 任一输入已无效则不能恢复，返回None。 |
| 61 | <code>    merged = []</code> | 分别合并货币和尺度：双方都声明且不同则冲突，否则用非空一方。 |
| 62 | <code>    for a, b in zip(left[:2], right[:2]):</code> | 分别合并货币和尺度：双方都声明且不同则冲突，否则用非空一方。 |
| 63 | <code>        if a and b and a != b:</code> | 分别合并货币和尺度：双方都声明且不同则冲突，否则用非空一方。 |
| 64 | <code>            return None</code> | 分别合并货币和尺度：双方都声明且不同则冲突，否则用非空一方。 |
| 65 | <code>        merged.append(a or b)</code> | 分别合并货币和尺度：双方都声明且不同则冲突，否则用非空一方。 |
| 66 | <code>    percent = left[2] or right[2]</code> | 百分比标志取逻辑或。 |
| 67 | <code>    if percent and any(merged):</code> | 合并后百分比与其他单位共存也拒绝。 |
| 68 | <code>        return None</code> | 合并后百分比与其他单位共存也拒绝。 |
| 69 | <code>    return (*merged, percent)</code> | 返回两个合并元组加百分比标志。 |
| 70 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 71 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 72 | <code>def _has_unit(unit):</code> | _has_unit要求签名有效且至少一部分明确；全空单位不够支撑证书。 |
| 73 | <code>    return unit is not None and any(unit)</code> | _has_unit要求签名有效且至少一部分明确；全空单位不够支撑证书。 |
| 74 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 75 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 76 | <code>def _amount_pattern(prefix):</code> | _amount_pattern(prefix)生成具命名分组的金额正则，prefix避免同一二元表达式的多金额分组重名。 |
| 77 | <code>    return (rf"(?P&lt;{prefix}currency&gt;[$£€]&#124;\bUSD\b&#124;\bGBP\b&#124;\bEUR\b)?[ \t]*"</code> | 金额前可选货币，限定水平空白；不跨换行吞句子。 |
| 78 | <code>            rf"(?P&lt;{prefix}value&gt;{_NUM})[ \t]*"</code> | 数值部分必需，用有限_NUM语法。 |
| 79 | <code>            rf"(?P&lt;{prefix}scale&gt;{_SCALE})?[ \t]*(?P&lt;{prefix}percent&gt;%&#124;percent\b)?")</code> | 数值后可选尺度及百分比，分组名带prefix。 |
| 80 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 81 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 82 | <code>def _amount(match, prefix):</code> | _amount将regex match的指定金额组转Decimal、单位和字符role范围；由来源单元格和回答断言共用。 |
| 83 | <code>    spans = [match.span(prefix + part) for part in ("currency", "scale", "percent")</code> | 收集实际出现的货币/尺度/百分比字符区间，缺省单位不造虚拟role。 |
| 84 | <code>             if match.group(prefix + part)]</code> | 收集实际出现的货币/尺度/百分比字符区间，缺省单位不造虚拟role。 |
| 85 | <code>    unit = _unit(" ".join(match.group(prefix + part) or ""</code> | 将出现的单位部分拼起来调用严格_unit解析。 |
| 86 | <code>                          for part in ("currency", "scale", "percent")))</code> | 将出现的单位部分拼起来调用严格_unit解析。 |
| 87 | <code>    literal = match.group(prefix + "value")</code> | 取原始数字字面及其在匹配文本中的绝对字符起点。 |
| 88 | <code>    start, _ = match.span(prefix + "value")</code> | 取原始数字字面及其在匹配文本中的绝对字符起点。 |
| 89 | <code>    magnitude = re.search(r"\d[\d,.]*", literal)</code> | 定位纯数值幅度范围，排除正负号/外围括号。 |
| 90 | <code>    sign_spans = [(start + i, start + i + 1) for i, char in enumerate(literal) if char in "+-()"]</code> | 单独记录显式+-()每个字符区间，使符号与幅度可以归到不同role。 |
| 91 | <code>    return {"value": _decimal(literal), "unit": unit, "literal": literal,</code> | 返回数值、单位、字面、幅度span、符号spans和单位spans。没有生成任何替换字符串。 |
| 92 | <code>            "value_span": (start + magnitude.start(), start + magnitude.end()),</code> | 返回数值、单位、字面、幅度span、符号spans和单位spans。没有生成任何替换字符串。 |
| 93 | <code>            "sign_spans": sign_spans, "unit_spans": spans}</code> | 返回数值、单位、字面、幅度span、符号spans和单位spans。没有生成任何替换字符串。 |
| 94 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 95 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 96 | <code>def _header_unit(text):</code> | _header_unit只接受完整单位或明确的末尾单位括号，不从任意标题猜单位。 |
| 97 | <code>    """Only a complete unit literal or an explicit terminal unit parenthesis."""</code> | _header_unit只接受完整单位或明确的末尾单位括号，不从任意标题猜单位。 |
| 98 | <code>    unit = _unit(text)</code> | 能作为完整_unit解析时直接返回，包括未说明的空签名。 |
| 99 | <code>    if unit is not None:</code> | 能作为完整_unit解析时直接返回，包括未说明的空签名。 |
| 100 | <code>        return unit</code> | 能作为完整_unit解析时直接返回，包括未说明的空签名。 |
| 101 | <code>    prefix = r"(?:(?:amounts?&#124;values?&#124;years? ended (?:january&#124;february&#124;march&#124;april&#124;may&#124;june&#124;july&#124;august&#124;september&#124;october&#124;november&#124;december) \d{1,2},?)\s*)?"</code> | 否则只允许amount/value或明确year-ended日期等有限表头前缀。 |
| 102 | <code>    match = re.fullmatch(prefix + r"\(\s*(?P&lt;unit&gt;(?:in\s+)?(?:[$£€]\s*)?(?:thousand&#124;million&#124;billion)s?)\s*\)",</code> | 完整匹配前缀+括号尺度后解析其中unit；其他标题返回None，不能把字段名称误当单位。 |
| 103 | <code>                         _name(text))</code> | 完整匹配前缀+括号尺度后解析其中unit；其他标题返回None，不能把字段名称误当单位。 |
| 104 | <code>    return _unit(match["unit"]) if match else None</code> | 完整匹配前缀+括号尺度后解析其中unit；其他标题返回None，不能把字段名称误当单位。 |
| 105 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 106 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 107 | <code>def _json_tables(source):</code> | _json_tables是生成器，扫描来源中以字面'[['开头的JSON表，并返回表起止、行值和每cell原文span；不支持所有可能JSON排版。 |
| 108 | <code>    """Decode JSON cells while retaining each cell's original source span."""</code> | _json_tables是生成器，扫描来源中以字面'[['开头的JSON表，并返回表起止、行值和每cell原文span；不支持所有可能JSON排版。 |
| 109 | <code>    decoder = json.JSONDecoder()</code> | 创建解码器并从字符0顺序扫描。 |
| 110 | <code>    cursor = 0</code> | 创建解码器并从字符0顺序扫描。 |
| 111 | <code>    while (start := source.find("[[", cursor)) &gt;= 0:</code> | 海象运算绑定下一处'[['；找不到时循环结束，不搜索重复cell文字定位来源。 |
| 112 | <code>        try:</code> | 该位置不能解码JSON则游标至少前移2并继续，保证不会在坏前缀上死循环。 |
| 113 | <code>            table, size = decoder.raw_decode(source[start:])</code> | 该位置不能解码JSON则游标至少前移2并继续，保证不会在坏前缀上死循环。 |
| 114 | <code>        except json.JSONDecodeError:</code> | 该位置不能解码JSON则游标至少前移2并继续，保证不会在坏前缀上死循环。 |
| 115 | <code>            cursor = start + 2</code> | 该位置不能解码JSON则游标至少前移2并继续，保证不会在坏前缀上死循环。 |
| 116 | <code>            continue</code> | 该位置不能解码JSON则游标至少前移2并继续，保证不会在坏前缀上死循环。 |
| 117 | <code>        cursor = start + size</code> | 解码成功后游标移至整个JSON值末尾。 |
| 118 | <code>        if not (isinstance(table, list) and table</code> | 只接受非空list且每行都是list，其他JSON即使合法也不是本语法表。 |
| 119 | <code>                and all(isinstance(row, list) for row in table)):</code> | 只接受非空list且每行都是list，其他JSON即使合法也不是本语法表。 |
| 120 | <code>            continue</code> | 只接受非空list且每行都是list，其他JSON即使合法也不是本语法表。 |
| 121 | <code>        # Recover scalar cell boundaries without searching for repeated values.</code> | 为恢复单元格边界重新顺序解码标量；position从外层左括号后开始，valid记录结构一致性。 |
| 122 | <code>        cells, position = [], start + 1</code> | 为恢复单元格边界重新顺序解码标量；position从外层左括号后开始，valid记录结构一致性。 |
| 123 | <code>        valid = True</code> | 为恢复单元格边界重新顺序解码标量；position从外层左括号后开始，valid记录结构一致性。 |
| 124 | <code>        for row in table:</code> | 逐行跳过结构间空白。 |
| 125 | <code>            while position &lt; cursor and source[position].isspace():</code> | 逐行跳过结构间空白。 |
| 126 | <code>                position += 1</code> | 逐行跳过结构间空白。 |
| 127 | <code>            if source[position] != "[":</code> | 期待每行以'['开始；不满足则放弃当前表的边界恢复。 |
| 128 | <code>                valid = False</code> | 期待每行以'['开始；不满足则放弃当前表的边界恢复。 |
| 129 | <code>                break</code> | 期待每行以'['开始；不满足则放弃当前表的边界恢复。 |
| 130 | <code>            position += 1</code> | 越过行括号并初始化当前行cell spans。 |
| 131 | <code>            spans = []</code> | 越过行括号并初始化当前行cell spans。 |
| 132 | <code>            for _ in row:</code> | 按已解析行cell个数逐个raw_decode，记录原始起止，再前移实际JSON字面长度；即使值重复也不会定位错cell。 |
| 133 | <code>                while source[position].isspace():</code> | 按已解析行cell个数逐个raw_decode，记录原始起止，再前移实际JSON字面长度；即使值重复也不会定位错cell。 |
| 134 | <code>                    position += 1</code> | 按已解析行cell个数逐个raw_decode，记录原始起止，再前移实际JSON字面长度；即使值重复也不会定位错cell。 |
| 135 | <code>                _, length = decoder.raw_decode(source[position:])</code> | 按已解析行cell个数逐个raw_decode，记录原始起止，再前移实际JSON字面长度；即使值重复也不会定位错cell。 |
| 136 | <code>                spans.append((position, position + length))</code> | 按已解析行cell个数逐个raw_decode，记录原始起止，再前移实际JSON字面长度；即使值重复也不会定位错cell。 |
| 137 | <code>                position += length</code> | 按已解析行cell个数逐个raw_decode，记录原始起止，再前移实际JSON字面长度；即使值重复也不会定位错cell。 |
| 138 | <code>                while source[position].isspace():</code> | 跳过单元格后的空白及可选逗号，准备下一个cell。 |
| 139 | <code>                    position += 1</code> | 跳过单元格后的空白及可选逗号，准备下一个cell。 |
| 140 | <code>                if source[position] == ",":</code> | 跳过单元格后的空白及可选逗号，准备下一个cell。 |
| 141 | <code>                    position += 1</code> | 跳过单元格后的空白及可选逗号，准备下一个cell。 |
| 142 | <code>            while source[position].isspace():</code> | 跳过尾部空白再越过已由JSON解析确认的行右括号。 |
| 143 | <code>                position += 1</code> | 跳过尾部空白再越过已由JSON解析确认的行右括号。 |
| 144 | <code>            position += 1  # closing row bracket</code> | 跳过尾部空白再越过已由JSON解析确认的行右括号。 |
| 145 | <code>            while position &lt; cursor and source[position].isspace():</code> | 跳过行间空白及行分隔逗号，游标仍限制在完整表JSON内。 |
| 146 | <code>                position += 1</code> | 跳过行间空白及行分隔逗号，游标仍限制在完整表JSON内。 |
| 147 | <code>            if position &lt; cursor and source[position] == ",":</code> | 跳过行间空白及行分隔逗号，游标仍限制在完整表JSON内。 |
| 148 | <code>                position += 1</code> | 跳过行间空白及行分隔逗号，游标仍限制在完整表JSON内。 |
| 149 | <code>            cells.append(spans)</code> | 保存当前行各cell字符区间，轴与rows对应。 |
| 150 | <code>        if valid:</code> | 只有边界恢复有效才yield表起止、解析值与spans，调用方继续扫描后续表。 |
| 151 | <code>            yield start, cursor, table, cells</code> | 只有边界恢复有效才yield表起止、解析值与spans，调用方继续扫描后续表。 |
| 152 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 153 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 154 | <code>def _source_scopes(source):</code> | _source_scopes将来源表编译为封闭年份/行/单位事实域，返回scopes与结构问题；由compile_proofs调用，不读取标签。 |
| 155 | <code>    scopes, problems = [], []</code> | 给各表顺序ID，并获取原文范围及cell spans。 |
| 156 | <code>    for table_id, (start, stop, rows, spans) in enumerate(_json_tables(source)):</code> | 给各表顺序ID，并获取原文范围及cell spans。 |
| 157 | <code>        # Header candidates belong to the structured prefix. A year-looking</code> | 注释说明年份header只在结构前缀找，不能把名为某行的数据金额2000误认成年份表头。 |
| 158 | <code>        # amount in a named data row must never become a second year header.</code> | 注释说明年份header只在结构前缀找，不能把名为某行的数据金额2000误认成年份表头。 |
| 159 | <code>        headers = [(i, {j: str(cell).strip() for j, cell in enumerate(row[1:], 1)</code> | 候选行必须首行、首cell空或明确单位标题；从后续cell只收完整四位年份，保存列号→年份。 |
| 160 | <code>                        if re.fullmatch(_YEAR, str(cell).strip())})</code> | 候选行必须首行、首cell空或明确单位标题；从后续cell只收完整四位年份，保存列号→年份。 |
| 161 | <code>                   for i, row in enumerate(rows)</code> | 候选行必须首行、首cell空或明确单位标题；从后续cell只收完整四位年份，保存列号→年份。 |
| 162 | <code>                   if row and (i == 0 or not str(row[0]).strip() or _has_unit(_header_unit(str(row[0]))))]</code> | 候选行必须首行、首cell空或明确单位标题；从后续cell只收完整四位年份，保存列号→年份。 |
| 163 | <code>        headers = [(i, years) for i, years in headers if years]</code> | 去掉没有年份的候选。 |
| 164 | <code>        if len(headers) != 1 or any(re.fullmatch(_amount_pattern("prior"), str(cell).strip(), re.I)</code> | 必须只有一个年份表头，且它之前不能已有可解析金额数据；否则记录missing_or_ambiguous_year_header。 |
| 165 | <code>                                    for row in rows[:headers[0][0]] for cell in row[1:]):</code> | 必须只有一个年份表头，且它之前不能已有可解析金额数据；否则记录missing_or_ambiguous_year_header。 |
| 166 | <code>            problems.append("missing_or_ambiguous_year_header")</code> | 必须只有一个年份表头，且它之前不能已有可解析金额数据；否则记录missing_or_ambiguous_year_header。 |
| 167 | <code>            scopes.append({"table": table_id, "span": [start, stop], "facts": {},</code> | 歧义表仍保留表ID/范围/subjects供后续unknown解释，但facts为空、complete=False，不假造可用事实。 |
| 168 | <code>                           "complete": False, "header_context": [], "units": set(),</code> | 歧义表仍保留表ID/范围/subjects供后续unknown解释，但facts为空、complete=False，不假造可用事实。 |
| 169 | <code>                           "header_context_spans": [],</code> | 歧义表仍保留表ID/范围/subjects供后续unknown解释，但facts为空、complete=False，不假造可用事实。 |
| 170 | <code>                           "subjects": {_name(row[0]) for row in rows if row and str(row[0]).strip()}})</code> | 歧义表仍保留表ID/范围/subjects供后续unknown解释，但facts为空、complete=False，不假造可用事实。 |
| 171 | <code>            continue</code> | 跳过该不完整scope的事实枚举。 |
| 172 | <code>        header_index, years = headers[0]</code> | 唯一表头确定后读其显式单位；普通字段名表头可以存在，但没有声明单位就用空签名，后续金额必须补明确单位。 |
| 173 | <code>        header_unit = _header_unit(str(rows[header_index][0]))</code> | 唯一表头确定后读其显式单位；普通字段名表头可以存在，但没有声明单位就用空签名，后续金额必须补明确单位。 |
| 174 | <code>        if header_unit is None:</code> | 唯一表头确定后读其显式单位；普通字段名表头可以存在，但没有声明单位就用空签名，后续金额必须补明确单位。 |
| 175 | <code>            # A field-name header is allowed; it does not declare a unit.</code> | 唯一表头确定后读其显式单位；普通字段名表头可以存在，但没有声明单位就用空签名，后续金额必须补明确单位。 |
| 176 | <code>            header_unit = ((), (), False)</code> | 唯一表头确定后读其显式单位；普通字段名表头可以存在，但没有声明单位就用空签名，后续金额必须补明确单位。 |
| 177 | <code>        column_units = dict.fromkeys(years, header_unit)</code> | 每年份列先继承表头单位，数据起点在表头下一行。 |
| 178 | <code>        first_data = header_index + 1</code> | 每年份列先继承表头单位，数据起点在表头下一行。 |
| 179 | <code>        while first_data &lt; len(rows):</code> | 识别紧随表头的单位行：所有年份列必须是显式单位，首cell只能空或单位；任一不符就认为真正数据开始。 |
| 180 | <code>            row = rows[first_data]</code> | 识别紧随表头的单位行：所有年份列必须是显式单位，首cell只能空或单位；任一不符就认为真正数据开始。 |
| 181 | <code>            if not all(j &lt; len(row) and _has_unit(_unit(str(row[j]))) for j in years):</code> | 识别紧随表头的单位行：所有年份列必须是显式单位，首cell只能空或单位；任一不符就认为真正数据开始。 |
| 182 | <code>                break</code> | 识别紧随表头的单位行：所有年份列必须是显式单位，首cell只能空或单位；任一不符就认为真正数据开始。 |
| 183 | <code>            if row and str(row[0]).strip() and not _has_unit(_unit(str(row[0]))):</code> | 识别紧随表头的单位行：所有年份列必须是显式单位，首cell只能空或单位；任一不符就认为真正数据开始。 |
| 184 | <code>                break</code> | 识别紧随表头的单位行：所有年份列必须是显式单位，首cell只能空或单位；任一不符就认为真正数据开始。 |
| 185 | <code>            for j in years:</code> | 逐年份列合并单位行，成功后向下推进数据起点，不混入facts。 |
| 186 | <code>                column_units[j] = _merge_unit(column_units[j], _unit(str(row[j])))</code> | 逐年份列合并单位行，成功后向下推进数据起点，不混入facts。 |
| 187 | <code>            first_data += 1</code> | 逐年份列合并单位行，成功后向下推进数据起点，不混入facts。 |
| 188 | <code>        facts = {}</code> | 初始facts空、complete=True；随后逐数据行检查，任何未知cell会让整个scope失去封闭性。 |
| 189 | <code>        complete = True</code> | 初始facts空、complete=True；随后逐数据行检查，任何未知cell会让整个scope失去封闭性。 |
| 190 | <code>        for i in range(first_data, len(rows)):</code> | 初始facts空、complete=True；随后逐数据行检查，任何未知cell会让整个scope失去封闭性。 |
| 191 | <code>            row = rows[i]</code> | 空行或包含dict/list/bool的行不属于数值语法，标不完整并跳过。 |
| 192 | <code>            if not row or any(isinstance(cell, (dict, list, bool)) for cell in row):</code> | 空行或包含dict/list/bool的行不属于数值语法，标不完整并跳过。 |
| 193 | <code>                complete = False</code> | 空行或包含dict/list/bool的行不属于数值语法，标不完整并跳过。 |
| 194 | <code>                continue</code> | 空行或包含dict/list/bool的行不属于数值语法，标不完整并跳过。 |
| 195 | <code>            label = _name(row[0])</code> | 规范化首cell为subject，匿名行用带行号tuple保留独立路径，不能与命名subject混淆。 |
| 196 | <code>            path = label if label else ("anonymous_row", i)</code> | 规范化首cell为subject，匿名行用带行号tuple保留独立路径，不能与命名subject混淆。 |
| 197 | <code>            for j, year in years.items():</code> | 每年份列都必须存在cell，缺列标不完整。 |
| 198 | <code>                if j &gt;= len(row):</code> | 每年份列都必须存在cell，缺列标不完整。 |
| 199 | <code>                    complete = False</code> | 每年份列都必须存在cell，缺列标不完整。 |
| 200 | <code>                    continue</code> | 每年份列都必须存在cell，缺列标不完整。 |
| 201 | <code>                literal = str(row[j]).strip()</code> | cell字符串必须完整匹配金额语法，不从混杂散文中抽一个数凑事实。 |
| 202 | <code>                match = re.fullmatch(_amount_pattern("cell"), literal, re.I)</code> | cell字符串必须完整匹配金额语法，不从混杂散文中抽一个数凑事实。 |
| 203 | <code>                if match is None:</code> | cell字符串必须完整匹配金额语法，不从混杂散文中抽一个数凑事实。 |
| 204 | <code>                    complete = False</code> | cell字符串必须完整匹配金额语法，不从混杂散文中抽一个数凑事实。 |
| 205 | <code>                    continue</code> | cell字符串必须完整匹配金额语法，不从混杂散文中抽一个数凑事实。 |
| 206 | <code>                amount = _amount(match, "cell")</code> | 解析金额、合并列单位；若没有明确兼容单位则整个scope标不完整，不推断美元/百万。 |
| 207 | <code>                unit = _merge_unit(column_units[j], amount["unit"])</code> | 解析金额、合并列单位；若没有明确兼容单位则整个scope标不完整，不推断美元/百万。 |
| 208 | <code>                if not _has_unit(unit):</code> | 解析金额、合并列单位；若没有明确兼容单位则整个scope标不完整，不推断美元/百万。 |
| 209 | <code>                    complete = False</code> | 解析金额、合并列单位；若没有明确兼容单位则整个scope标不完整，不推断美元/百万。 |
| 210 | <code>                    continue</code> | 解析金额、合并列单位；若没有明确兼容单位则整个scope标不完整，不推断美元/百万。 |
| 211 | <code>                key = (path, year, unit)</code> | 事实key=(行路径,年份,单位)，不只用金额或年份做key。 |
| 212 | <code>                fact = facts.setdefault(key, {"subject": label, "path": path, "year": year,</code> | 同key聚集subject/path/year/unit及values映射；重复key可以揭示矛盾值。 |
| 213 | <code>                                             "unit": unit, "values": {}})</code> | 同key聚集subject/path/year/unit及values映射；重复key可以揭示矛盾值。 |
| 214 | <code>                provenance = {"table": table_id, "table_span": [start, stop], "row": i,</code> | 每个值保留table/行列ID、表/行名/值/年份header原文span和literal，证书可回指实际来源cell。 |
| 215 | <code>                              "column": j, "row_span": list(spans[i][0]),</code> | 每个值保留table/行列ID、表/行名/值/年份header原文span和literal，证书可回指实际来源cell。 |
| 216 | <code>                              "value_span": list(spans[i][j]),</code> | 每个值保留table/行列ID、表/行名/值/年份header原文span和literal，证书可回指实际来源cell。 |
| 217 | <code>                              "header_span": list(spans[header_index][j]),</code> | 每个值保留table/行列ID、表/行名/值/年份header原文span和literal，证书可回指实际来源cell。 |
| 218 | <code>                              "literal": literal}</code> | 每个值保留table/行列ID、表/行名/值/年份header原文span和literal，证书可回指实际来源cell。 |
| 219 | <code>                fact["values"].setdefault(amount["value"], []).append(provenance)</code> | 同Decimal值的多个出处累积列表，不覆盖 provenance；不同值同key会被后续判矛盾。 |
| 220 | <code>        scopes.append({"table": table_id, "span": [start, stop], "facts": facts,</code> | 完整scope保存facts、封闭性、结构表头上下文与spans、数据subjects和实际已解析单位集合。 |
| 221 | <code>                       "complete": complete,</code> | 完整scope保存facts、封闭性、结构表头上下文与spans、数据subjects和实际已解析单位集合。 |
| 222 | <code>                       "header_context": rows[:first_data],</code> | 完整scope保存facts、封闭性、结构表头上下文与spans、数据subjects和实际已解析单位集合。 |
| 223 | <code>                       "header_context_spans": spans[:first_data],</code> | 完整scope保存facts、封闭性、结构表头上下文与spans、数据subjects和实际已解析单位集合。 |
| 224 | <code>                       "subjects": {_name(row[0]) for row in rows[first_data:] if row and str(row[0]).strip()},</code> | 完整scope保存facts、封闭性、结构表头上下文与spans、数据subjects和实际已解析单位集合。 |
| 225 | <code>                       "units": {key[2] for key in facts}})</code> | 完整scope保存facts、封闭性、结构表头上下文与spans、数据subjects和实际已解析单位集合。 |
| 226 | <code>    return scopes, problems</code> | 返回所有scope和结构问题，不把'未解析'默认成'不支持'。 |
| 227 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 228 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 229 | <code>def _anchors(response, scopes):</code> | _anchors在回答中按有限来源subjects找字面锚点，返回字符起止及subject；不做开放实体抽取。 |
| 230 | <code>    labels = {subject for scope in scopes for subject in scope["subjects"]}</code> | 合并来源subject去重并排序，保证遍历确定。 |
| 231 | <code>    matches = []</code> | 合并来源subject去重并排序，保证遍历确定。 |
| 232 | <code>    for label in sorted(labels):</code> | 合并来源subject去重并排序，保证遍历确定。 |
| 233 | <code>        pattern = r"(?&lt;![\w-])" + r"\s+".join(re.escape(s) for s in label.split()) + r"(?![\w-])"</code> | 将每个subject转为转义字面正则，词间允许空白，边界不能嵌在更长单词/连字符实体内部。 |
| 234 | <code>        for match in re.finditer(pattern, response, re.I):</code> | 不区分大小写找全部匹配，记录回答字符坐标。 |
| 235 | <code>            matches.append((match.start(), match.end(), label))</code> | 不区分大小写找全部匹配，记录回答字符坐标。 |
| 236 | <code>    return [m for m in matches if not any(a &lt;= m[0] and b &gt;= m[1] and b-a &gt; m[1]-m[0]</code> | 若某锚点完全被更长锚点覆盖就去掉短者，避免把复合行名误匹配成子名字。 |
| 237 | <code>                                         for a, b, _ in matches)]</code> | 若某锚点完全被更长锚点覆盖就去掉短者，避免把复合行名误匹配成子名字。 |
| 238 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 239 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 240 | <code>def _claims(response, scopes):</code> | _claims从字面锚点编译允许的直接/二元断言，返回regex matches与schema/spans；未识别句子不会强行解析。 |
| 241 | <code>    claims = []</code> | 句尾必须是非小数句点、分号、换行或末尾；逐subject锚点准备金额组，防止断言继续被未解析内容改义。 |
| 242 | <code>    endpoint = r"(?=[ \t]*(?:\.(?!\d)&#124;;&#124;\n&#124;$))"</code> | 句尾必须是非小数句点、分号、换行或末尾；逐subject锚点准备金额组，防止断言继续被未解析内容改义。 |
| 243 | <code>    for start, stop, subject in _anchors(response, scopes):</code> | 句尾必须是非小数句点、分号、换行或末尾；逐subject锚点准备金额组，防止断言继续被未解析内容改义。 |
| 244 | <code>        amount = _amount_pattern("a")</code> | 句尾必须是非小数句点、分号、换行或末尾；逐subject锚点准备金额组，防止断言继续被未解析内容改义。 |
| 245 | <code>        patterns = [</code> | 第一直接语法：ROW in/for/during YEAR was/were/is/are/totaled AMOUNT。 |
| 246 | <code>            ("direct", rf"[ \t]+(?:in&#124;for&#124;during)[ \t]+(?P&lt;year&gt;{_YEAR})[ \t]+(?:was&#124;were&#124;is&#124;are&#124;totaled&#124;totalled)[ \t]+{amount}{endpoint}"),</code> | 第一直接语法：ROW in/for/during YEAR was/were/is/are/totaled AMOUNT。 |
| 247 | <code>            ("direct", rf"[ \t]+(?:was&#124;were&#124;is&#124;are&#124;totaled&#124;totalled)[ \t]+{amount}[ \t]+(?:in&#124;for&#124;during)[ \t]+(?P&lt;year&gt;{_YEAR}){endpoint}"),</code> | 第二直接语法：ROW was AMOUNT in YEAR，明确年份位置。 |
| 248 | <code>            ("binary", rf"[ \t]*:[ \t]*{_amount_pattern('left')}[ \t]+in[ \t]+(?P&lt;leftyear&gt;{_YEAR})[ \t]*(?P&lt;operator&gt;[+-])[ \t]*{_amount_pattern('right')}[ \t]+in[ \t]+(?P&lt;rightyear&gt;{_YEAR})[ \t]*=[ \t]*{_amount_pattern('result')}{endpoint}"),</code> | 二元语法：ROW: AMOUNT in YEAR +/- AMOUNT in YEAR = AMOUNT；操作符和三个金额分别命名。 |
| 249 | <code>        ]</code> | 每模式必须从subject末尾紧接匹配，不在后续任意位置搜一段看起来像断言的文字。 |
| 250 | <code>        for schema, pattern in patterns:</code> | 每模式必须从subject末尾紧接匹配，不在后续任意位置搜一段看起来像断言的文字。 |
| 251 | <code>            match = re.compile(pattern, re.I).match(response, stop)</code> | 每模式必须从subject末尾紧接匹配，不在后续任意位置搜一段看起来像断言的文字。 |
| 252 | <code>            if match:</code> | 每模式必须从subject末尾紧接匹配，不在后续任意位置搜一段看起来像断言的文字。 |
| 253 | <code>                claims.append({"schema": schema, "subject": subject, "subject_span": (start, stop),</code> | 记录schema、subject范围、整断言范围和match对象供role解析。 |
| 254 | <code>                               "span": (start, match.end()), "match": match})</code> | 记录schema、subject范围、整断言范围和match对象供role解析。 |
| 255 | <code>        # The year-first schema is anchored backward, without consuming prose.</code> | 额外支持year-first：在subject之前只匹配紧邻的In YEAR等有限前缀，之后仍用严格was AMOUNT语法。 |
| 256 | <code>        before = re.search(rf"\b(?:In&#124;For&#124;During)[ \t]+(?P&lt;year&gt;{_YEAR}),?[ \t]+(?:the[ \t]+(?:amount[ \t]+of[ \t]+)?)?$",</code> | 额外支持year-first：在subject之前只匹配紧邻的In YEAR等有限前缀，之后仍用严格was AMOUNT语法。 |
| 257 | <code>                           response[:start], re.I)</code> | 额外支持year-first：在subject之前只匹配紧邻的In YEAR等有限前缀，之后仍用严格was AMOUNT语法。 |
| 258 | <code>        after = re.compile(rf"[ \t]+(?:was&#124;were&#124;is&#124;are&#124;totaled&#124;totalled)[ \t]+{amount}{endpoint}", re.I).match(response, stop)</code> | 额外支持year-first：在subject之前只匹配紧邻的In YEAR等有限前缀，之后仍用严格was AMOUNT语法。 |
| 259 | <code>        if before and after:</code> | 两侧都匹配才形成直接断言，年份match单独保存以便正确字符role。 |
| 260 | <code>            claims.append({"schema": "direct", "subject": subject, "subject_span": (start, stop),</code> | 两侧都匹配才形成直接断言，年份match单独保存以便正确字符role。 |
| 261 | <code>                           "span": (before.start(), after.end()), "match": after,</code> | 两侧都匹配才形成直接断言，年份match单独保存以便正确字符role。 |
| 262 | <code>                           "year_match": before})</code> | 两侧都匹配才形成直接断言，年份match单独保存以便正确字符role。 |
| 263 | <code>    unique = {}</code> | 按schema/subject-span/整span去重，返回唯一声明列表。 |
| 264 | <code>    for claim in claims:</code> | 按schema/subject-span/整span去重，返回唯一声明列表。 |
| 265 | <code>        unique[(claim["schema"], claim["subject_span"], claim["span"])] = claim</code> | 按schema/subject-span/整span去重，返回唯一声明列表。 |
| 266 | <code>    return list(unique.values())</code> | 按schema/subject-span/整span去重，返回唯一声明列表。 |
| 267 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 268 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 269 | <code>def _assertion_scope(response, claims):</code> | _assertion_scope要求整个回答由有限声明句组成；任意前言/跟进可能引入否定、引用、假设，因此不能只抽到一句就当真实断言。 |
| 270 | <code>    """Require the whole response to consist of finite declarative clauses.</code> | _assertion_scope要求整个回答由有限声明句组成；任意前言/跟进可能引入否定、引用、假设，因此不能只抽到一句就当真实断言。 |
| 271 | <code> </code> | _assertion_scope要求整个回答由有限声明句组成；任意前言/跟进可能引入否定、引用、假设，因此不能只抽到一句就当真实断言。 |
| 272 | <code>    No arbitrary preamble or follow-up prose is permitted: it could retract,</code> | _assertion_scope要求整个回答由有限声明句组成；任意前言/跟进可能引入否定、引用、假设，因此不能只抽到一句就当真实断言。 |
| 273 | <code>    quote, hypothesize or otherwise change the scope of an apparent assertion.</code> | _assertion_scope要求整个回答由有限声明句组成；任意前言/跟进可能引入否定、引用、假设，因此不能只抽到一句就当真实断言。 |
| 274 | <code>    """</code> | _assertion_scope要求整个回答由有限声明句组成；任意前言/跟进可能引入否定、引用、假设，因此不能只抽到一句就当真实断言。 |
| 275 | <code>    if (re.search(r'["“”«»`?]&#124;(?&lt;!\w)[\'‘’]&#124;[\'‘’](?!\w)', response)</code> | 出现引号/疑问/孤立撇号或否定、模态、假设、驳斥等词就整体弃权；这是保守词面门槛，不是普遍语义充分性证明。 |
| 276 | <code>            or re.search(r"\b(?:if&#124;unless&#124;suppose&#124;supposed&#124;assume&#124;assumed&#124;assuming&#124;assumptions?&#124;hypothetically&#124;hypothetical&#124;hypotheses&#124;hypothesis&#124;imagine&#124;pretend&#124;whether&#124;false&#124;untrue&#124;incorrect&#124;erroneous&#124;counterfactual&#124;conditional&#124;reject\w*&#124;den(?:y&#124;ies&#124;ied&#124;ial)&#124;not&#124;never&#124;may&#124;might&#124;would&#124;could&#124;should&#124;allegedly&#124;reportedly)\b", response, re.I)):</code> | 出现引号/疑问/孤立撇号或否定、模态、假设、驳斥等词就整体弃权；这是保守词面门槛，不是普遍语义充分性证明。 |
| 277 | <code>        return False</code> | 出现引号/疑问/孤立撇号或否定、模态、假设、驳斥等词就整体弃权；这是保守词面门槛，不是普遍语义充分性证明。 |
| 278 | <code>    regions = []</code> | 对每claim从此前最近句界找frame起点，提取断言前缀。 |
| 279 | <code>    for claim in claims:</code> | 对每claim从此前最近句界找frame起点，提取断言前缀。 |
| 280 | <code>        start, stop = claim["span"]</code> | 对每claim从此前最近句界找frame起点，提取断言前缀。 |
| 281 | <code>        boundaries = [m.end() for m in re.finditer(r"[.!?;][ \t]+&#124;\n", response[:start])]</code> | 对每claim从此前最近句界找frame起点，提取断言前缀。 |
| 282 | <code>        frame_start = boundaries[-1] if boundaries else 0</code> | 对每claim从此前最近句界找frame起点，提取断言前缀。 |
| 283 | <code>        prefix = response[frame_start:start]</code> | 对每claim从此前最近句界找frame起点，提取断言前缀。 |
| 284 | <code>        if re.fullmatch(r"[ \t]*(?:(?:[-*•]&#124;\d{1,2}[.)])[ \t]+)?(?:the[ \t]+(?:amount[ \t]+of[ \t]+)?)?",</code> | 前缀只允许空白、有限列表标记和the amount of等明确装饰，其他散文改变作用域就拒绝。 |
| 285 | <code>                        prefix, re.I) is None:</code> | 前缀只允许空白、有限列表标记和the amount of等明确装饰，其他散文改变作用域就拒绝。 |
| 286 | <code>            return False</code> | 前缀只允许空白、有限列表标记和the amount of等明确装饰，其他散文改变作用域就拒绝。 |
| 287 | <code>        regions.append((frame_start, stop))</code> | 收集声明覆盖范围并按顺序检查整回答，最后加一个末尾哨兵处理尾巴。 |
| 288 | <code>    position = 0</code> | 收集声明覆盖范围并按顺序检查整回答，最后加一个末尾哨兵处理尾巴。 |
| 289 | <code>    for start, stop in sorted(regions) + [(len(response), len(response))]:</code> | 收集声明覆盖范围并按顺序检查整回答，最后加一个末尾哨兵处理尾巴。 |
| 290 | <code>        gap = response[position:start]</code> | 未覆盖间隙只能空白/分号/句点或单独列表编号，否则有未知散文，整体作用域无保证。 |
| 291 | <code>        gap = re.sub(r"(?m)^[ \t]*\d{1,2}[.)][ \t]+$", "", gap)</code> | 未覆盖间隙只能空白/分号/句点或单独列表编号，否则有未知散文，整体作用域无保证。 |
| 292 | <code>        if re.fullmatch(r"\s*(?:[.;]\s*)?", gap) is None:</code> | 未覆盖间隙只能空白/分号/句点或单独列表编号，否则有未知散文，整体作用域无保证。 |
| 293 | <code>            return False</code> | 未覆盖间隙只能空白/分号/句点或单独列表编号，否则有未知散文，整体作用域无保证。 |
| 294 | <code>        position = max(position, stop)</code> | 推进已覆盖字符游标，重叠声明不使游标回退。 |
| 295 | <code>    return bool(regions)</code> | 只有至少一个声明且所有间隙合法才返回True。 |
| 296 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 297 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 298 | <code>def _resolved_unit(observed, scope):</code> | _resolved_unit在scope的已知单位中找与observed可合并成同单位的候选，只有唯一候选才返回；缺失/歧义返回None，不任选。 |
| 299 | <code>    possibilities = [unit for unit in scope["units"] if _merge_unit(observed, unit) == unit]</code> | _resolved_unit在scope的已知单位中找与observed可合并成同单位的候选，只有唯一候选才返回；缺失/歧义返回None，不任选。 |
| 300 | <code>    return possibilities[0] if len(possibilities) == 1 else None</code> | _resolved_unit在scope的已知单位中找与observed可合并成同单位的候选，只有唯一候选才返回；缺失/歧义返回None，不任选。 |
| 301 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 302 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 303 | <code>def _terms(terms):</code> | _terms规范化单调DNF：输入role序号项，返回无重复、无被包含冗余项的最小修复集合列表。 |
| 304 | <code>    canonical = sorted({tuple(sorted(set(term))) for term in terms}, key=lambda t: (len(t), t))</code> | 每term内部去重排序，全体去重；嵌套lambda按(term长度,字典序)排序，小集合先处理。 |
| 305 | <code>    result = []</code> | 若先前已保留term是当前term子集，当前更大修复集合不会改变DNF事件，丢弃；否则保留。 |
| 306 | <code>    for term in canonical:</code> | 若先前已保留term是当前term子集，当前更大修复集合不会改变DNF事件，丢弃；否则保留。 |
| 307 | <code>        if not any(set(previous).issubset(term) for previous in result):</code> | 若先前已保留term是当前term子集，当前更大修复集合不会改变DNF事件，丢弃；否则保留。 |
| 308 | <code>            result.append(term)</code> | 若先前已保留term是当前term子集，当前更大修复集合不会改变DNF事件，丢弃；否则保留。 |
| 309 | <code>    return [list(term) for term in result]</code> | 转回列表便于JSON持久化。 |
| 310 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 311 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 312 | <code>def _role_tokens(roles, response, offsets, selected):</code> | _role_tokens把每role字符spans映射到selected有效token轴，返回每role局部索引列表或None；不能将跨role token硬拆。 |
| 313 | <code>    result, ownership = [], {}</code> | ownership追踪每个有效token唯一归属，逐role初始化索引。 |
| 314 | <code>    for role_index, role in enumerate(roles):</code> | ownership追踪每个有效token唯一归属，逐role初始化索引。 |
| 315 | <code>        spans = role["spans"]</code> | ownership追踪每个有效token唯一归属，逐role初始化索引。 |
| 316 | <code>        indices = []</code> | ownership追踪每个有效token唯一归属，逐role初始化索引。 |
| 317 | <code>        for local, original in enumerate(selected):</code> | 逐selected原token取offset，只处理与当前role字符范围有交集者。 |
| 318 | <code>            left, right = offsets[original]</code> | 逐selected原token取offset，只处理与当前role字符范围有交集者。 |
| 319 | <code>            if not any(left &lt; b and right &gt; a for a, b in spans):</code> | 逐selected原token取offset，只处理与当前role字符范围有交集者。 |
| 320 | <code>                continue</code> | 逐selected原token取offset，只处理与当前role字符范围有交集者。 |
| 321 | <code>            if any(not response[p].isspace() and not any(a &lt;= p &lt; b for a, b in spans)</code> | 如果该token还覆盖role以外的非空白字符，映射不唯一，整体返回None。 |
| 322 | <code>                   for p in range(int(left), int(right))):</code> | 如果该token还覆盖role以外的非空白字符，映射不唯一，整体返回None。 |
| 323 | <code>                return None</code> | 如果该token还覆盖role以外的非空白字符，映射不唯一，整体返回None。 |
| 324 | <code>            if local in ownership and ownership[local] != role_index:</code> | 同token若已属别的role则拒绝，否则登记role并保留其在有效轴的local索引。 |
| 325 | <code>                return None</code> | 同token若已属别的role则拒绝，否则登记role并保留其在有效轴的local索引。 |
| 326 | <code>            ownership[local] = role_index</code> | 同token若已属别的role则拒绝，否则登记role并保留其在有效轴的local索引。 |
| 327 | <code>            indices.append(local)</code> | 同token若已属别的role则拒绝，否则登记role并保留其在有效轴的local索引。 |
| 328 | <code>        for a, b in spans:</code> | 反向检查role每个非空白字符确实被选中token覆盖，防止过滤/边界造成证书指向未评分字符。 |
| 329 | <code>            if any(not response[p].isspace() and not any(offsets[selected[t], 0] &lt;= p &lt; offsets[selected[t], 1]</code> | 反向检查role每个非空白字符确实被选中token覆盖，防止过滤/边界造成证书指向未评分字符。 |
| 330 | <code>                                                       for t in indices)</code> | 反向检查role每个非空白字符确实被选中token覆盖，防止过滤/边界造成证书指向未评分字符。 |
| 331 | <code>                   for p in range(a, b)):</code> | 反向检查role每个非空白字符确实被选中token覆盖，防止过滤/边界造成证书指向未评分字符。 |
| 332 | <code>                return None</code> | 反向检查role每个非空白字符确实被选中token覆盖，防止过滤/边界造成证书指向未评分字符。 |
| 333 | <code>        if not indices:</code> | role没有任何可投影token也拒绝；全部role成功才返回同序列表。 |
| 334 | <code>            return None</code> | role没有任何可投影token也拒绝；全部role成功才返回同序列表。 |
| 335 | <code>        result.append(indices)</code> | role没有任何可投影token也拒绝；全部role成功才返回同序列表。 |
| 336 | <code>    return result</code> | role没有任何可投影token也拒绝；全部role成功才返回同序列表。 |
| 337 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 338 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 339 | <code>def _numeric_roles(name, amount):</code> | _numeric_roles建立数值幅度role，只有出现显式符号字符时另建_sign role；没有符号不能造一个不存在的token作为修复对象。 |
| 340 | <code>    roles = [{"name": name, "spans": [amount["value_span"]]}]</code> | _numeric_roles建立数值幅度role，只有出现显式符号字符时另建_sign role；没有符号不能造一个不存在的token作为修复对象。 |
| 341 | <code>    if amount["sign_spans"]:</code> | _numeric_roles建立数值幅度role，只有出现显式符号字符时另建_sign role；没有符号不能造一个不存在的token作为修复对象。 |
| 342 | <code>        roles.append({"name": name + "_sign", "spans": amount["sign_spans"]})</code> | _numeric_roles建立数值幅度role，只有出现显式符号字符时另建_sign role；没有符号不能造一个不存在的token作为修复对象。 |
| 343 | <code>    return roles</code> | _numeric_roles建立数值幅度role，只有出现显式符号字符时另建_sign role；没有符号不能造一个不存在的token作为修复对象。 |
| 344 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 345 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 346 | <code>def _numeric_repairs(name, observed, witness, role_indices):</code> | _numeric_repairs对观察金额与候选见证Decimal比较，返回须修改的role序号及无法投影的隐式符号变化标志。 |
| 347 | <code>    """Explicit sign and magnitude are separate atoms; absent signs abstain."""</code> | _numeric_repairs对观察金额与候选见证Decimal比较，返回须修改的role序号及无法投影的隐式符号变化标志。 |
| 348 | <code>    magnitude_changed = observed["value"].copy_abs() != witness.copy_abs()</code> | 幅度不等就需要value role；符号比较独立于幅度。 |
| 349 | <code>    sign_changed = (observed["value"] &lt; 0) != (witness &lt; 0)</code> | 幅度不等就需要value role；符号比较独立于幅度。 |
| 350 | <code>    repairs = [role_indices[name]] if magnitude_changed else []</code> | 幅度不等就需要value role；符号比较独立于幅度。 |
| 351 | <code>    if sign_changed and observed["sign_spans"]:</code> | 符号不同且有显式sign span才加入sign role；若没有显式符号则标unprojectable，后续保守弃权。 |
| 352 | <code>        repairs.append(role_indices[name + "_sign"])</code> | 符号不同且有显式sign span才加入sign role；若没有显式符号则标unprojectable，后续保守弃权。 |
| 353 | <code>    return repairs, sign_changed and not observed["sign_spans"]</code> | 符号不同且有显式sign span才加入sign role；若没有显式符号则标unprojectable，后续保守弃权。 |
| 354 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 355 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 356 | <code>def _arithmetic(left, right, operator):</code> | _arithmetic在局部Decimal context按最大整数位和最小小数指数设置足够精度，防止默认精度舍入金额运算。 |
| 357 | <code>    with localcontext() as context:</code> | _arithmetic在局部Decimal context按最大整数位和最小小数指数设置足够精度，防止默认精度舍入金额运算。 |
| 358 | <code>        context.prec = (max(left.adjusted(), right.adjusted(), 0)</code> | _arithmetic在局部Decimal context按最大整数位和最小小数指数设置足够精度，防止默认精度舍入金额运算。 |
| 359 | <code>                        - min(left.as_tuple().exponent, right.as_tuple().exponent, 0) + 3)</code> | _arithmetic在局部Decimal context按最大整数位和最小小数指数设置足够精度，防止默认精度舍入金额运算。 |
| 360 | <code>        return left + right if operator == "+" else left - right</code> | 只执行有限语法允许的加法或减法，不求值任意表达式。 |
| 361 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 362 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 363 | <code>def _plausibly_rounded(value, observed):</code> | _plausibly_rounded判断表中精确值与显示更低精度金额的差异是否可能只是量化，用来弃权而非判错误。 |
| 364 | <code>    """Visible lower precision leaves ordinary quantization unresolved."""</code> | _plausibly_rounded判断表中精确值与显示更低精度金额的差异是否可能只是量化，用来弃权而非判错误。 |
| 365 | <code>    if value == observed:</code> | 相等不算舍入问题；否则只有observed小数指数更粗且绝对差小于一个显示量子时返回True。这个门槛是保守语法规则。 |
| 366 | <code>        return False</code> | 相等不算舍入问题；否则只有observed小数指数更粗且绝对差小于一个显示量子时返回True。这个门槛是保守语法规则。 |
| 367 | <code>    quantum = Decimal(1).scaleb(observed.as_tuple().exponent)</code> | 相等不算舍入问题；否则只有observed小数指数更粗且绝对差小于一个显示量子时返回True。这个门槛是保守语法规则。 |
| 368 | <code>    return (observed.as_tuple().exponent &gt; value.as_tuple().exponent</code> | 相等不算舍入问题；否则只有observed小数指数更粗且绝对差小于一个显示量子时返回True。这个门槛是保守语法规则。 |
| 369 | <code>            and _arithmetic(value, observed, "-").copy_abs() &lt; quantum)</code> | 相等不算舍入问题；否则只有observed小数指数更粗且绝对差小于一个显示量子时返回True。这个门槛是保守语法规则。 |
| 370 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 371 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 372 | <code>def _evaluate(claim, scopes):</code> | _evaluate是声明×封闭来源scope的核心：返回roles、最小repair terms、见证trace、状态与原因；不修改分数。 |
| 373 | <code>    match = claim["match"]</code> | 取regex match/subject，并建立主语字符role。 |
| 374 | <code>    subject = claim["subject"]</code> | 取regex match/subject，并建立主语字符role。 |
| 375 | <code>    roles = [{"name": "subject", "spans": [claim["subject_span"]]}]</code> | 取regex match/subject，并建立主语字符role。 |
| 376 | <code>    if claim["schema"] == "direct":</code> | direct模式取正确年份match、金额、year/value/sign角色与单位spans，保留原字符位置。 |
| 377 | <code>        year_match = claim.get("year_match", match)</code> | direct模式取正确年份match、金额、year/value/sign角色与单位spans，保留原字符位置。 |
| 378 | <code>        year = year_match["year"]</code> | direct模式取正确年份match、金额、year/value/sign角色与单位spans，保留原字符位置。 |
| 379 | <code>        amount = _amount(match, "a")</code> | direct模式取正确年份match、金额、year/value/sign角色与单位spans，保留原字符位置。 |
| 380 | <code>        roles += [{"name": "year", "spans": [year_match.span("year")]}]</code> | direct模式取正确年份match、金额、year/value/sign角色与单位spans，保留原字符位置。 |
| 381 | <code>        roles += _numeric_roles("value", amount)</code> | direct模式取正确年份match、金额、year/value/sign角色与单位spans，保留原字符位置。 |
| 382 | <code>        units = amount["unit_spans"]</code> | direct模式取正确年份match、金额、year/value/sign角色与单位spans，保留原字符位置。 |
| 383 | <code>    else:</code> | binary模式分别解析left/right/result，建立两年份、operator、各金额幅度/符号及共享unit角色候选。 |
| 384 | <code>        left, right, result = [_amount(match, prefix) for prefix in ("left", "right", "result")]</code> | binary模式分别解析left/right/result，建立两年份、operator、各金额幅度/符号及共享unit角色候选。 |
| 385 | <code>        roles += [{"name": "left_year", "spans": [match.span("leftyear")]}]</code> | binary模式分别解析left/right/result，建立两年份、operator、各金额幅度/符号及共享unit角色候选。 |
| 386 | <code>        roles += _numeric_roles("left_value", left)</code> | binary模式分别解析left/right/result，建立两年份、operator、各金额幅度/符号及共享unit角色候选。 |
| 387 | <code>        roles += [{"name": "operator", "spans": [match.span("operator")]},</code> | binary模式分别解析left/right/result，建立两年份、operator、各金额幅度/符号及共享unit角色候选。 |
| 388 | <code>                  {"name": "right_year", "spans": [match.span("rightyear")]}]</code> | binary模式分别解析left/right/result，建立两年份、operator、各金额幅度/符号及共享unit角色候选。 |
| 389 | <code>        roles += _numeric_roles("right_value", right) + _numeric_roles("result", result)</code> | binary模式分别解析left/right/result，建立两年份、operator、各金额幅度/符号及共享unit角色候选。 |
| 390 | <code>        units = left["unit_spans"] + right["unit_spans"] + result["unit_spans"]</code> | binary模式分别解析left/right/result，建立两年份、operator、各金额幅度/符号及共享unit角色候选。 |
| 391 | <code>        # Unsigned parentheses could mean accounting negation or math grouping.</code> | 无显式正负号的括号操作数可表示会计负数或数学分组，二元语境下不能任选，返回unknown。 |
| 392 | <code>        if any(re.fullmatch(r"\(\d[\d,.]*\)", match[prefix + "value"])</code> | 无显式正负号的括号操作数可表示会计负数或数学分组，二元语境下不能任选，返回unknown。 |
| 393 | <code>               for prefix in ("left", "right", "result")):</code> | 无显式正负号的括号操作数可表示会计负数或数学分组，二元语境下不能任选，返回unknown。 |
| 394 | <code>            return roles, [], [], "unknown", "ambiguous_parenthesized_operand"</code> | 无显式正负号的括号操作数可表示会计负数或数学分组，二元语境下不能任选，返回unknown。 |
| 395 | <code>    if units:</code> | 只有回答真的出现单位字面才创建unit角色；省略单位不造虚拟位置。 |
| 396 | <code>        roles.append({"name": "unit", "spans": units})</code> | 只有回答真的出现单位字面才创建unit角色；省略单位不造虚拟位置。 |
| 397 | <code>    if not claim["assertion_scope"]:</code> | 整回答不满足声明作用域门槛则unknown，禁止脱离引用/否定语境证明一个数字错。 |
| 398 | <code>        return roles, [], [], "unknown", "unresolved_assertion_scope"</code> | 整回答不满足声明作用域门槛则unknown，禁止脱离引用/否定语境证明一个数字错。 |
| 399 | <code>    role_indices = {role["name"]: i for i, role in enumerate(roles)}</code> | 映射role名到序号，准备来源解释、全修复项、见证trace及隐式符号不可投影标志。 |
| 400 | <code>    interpretations, all_terms, witness_traces = [], [], []</code> | 映射role名到序号，准备来源解释、全修复项、见证trace及隐式符号不可投影标志。 |
| 401 | <code>    unprojectable_sign = False</code> | 映射role名到序号，准备来源解释、全修复项、见证trace及隐式符号不可投影标志。 |
| 402 | <code>    candidates = [scope for scope in scopes if subject in scope["subjects"]]</code> | 只检查含该subject的scope；任一个候选表不完整就不能声称封闭穷尽，立即unknown。 |
| 403 | <code>    for scope in candidates:</code> | 只检查含该subject的scope；任一个候选表不完整就不能声称封闭穷尽，立即unknown。 |
| 404 | <code>        if not scope["complete"]:</code> | 只检查含该subject的scope；任一个候选表不完整就不能声称封闭穷尽，立即unknown。 |
| 405 | <code>            return roles, [], [], "unknown", "incomplete_table_scope"</code> | 只检查含该subject的scope；任一个候选表不完整就不能声称封闭穷尽，立即unknown。 |
| 406 | <code>        observed_units = [amount["unit"]] if claim["schema"] == "direct" else [a["unit"] for a in (left, right, result)]</code> | direct一个、binary三个观察单位都必须在scope内唯一解析且一致；否则missing_or_ambiguous_unit，不跨尺度强算。 |
| 407 | <code>        resolved = [_resolved_unit(unit, scope) for unit in observed_units]</code> | direct一个、binary三个观察单位都必须在scope内唯一解析且一致；否则missing_or_ambiguous_unit，不跨尺度强算。 |
| 408 | <code>        if any(unit is None for unit in resolved) or len(set(resolved)) != 1:</code> | direct一个、binary三个观察单位都必须在scope内唯一解析且一致；否则missing_or_ambiguous_unit，不跨尺度强算。 |
| 409 | <code>            return roles, [], [], "unknown", "missing_or_ambiguous_unit"</code> | direct一个、binary三个观察单位都必须在scope内唯一解析且一致；否则missing_or_ambiguous_unit，不跨尺度强算。 |
| 410 | <code>        unit = resolved[0]</code> | direct一个、binary三个观察单位都必须在scope内唯一解析且一致；否则missing_or_ambiguous_unit，不跨尺度强算。 |
| 411 | <code>        observed_years = [year] if claim["schema"] == "direct" else [match["leftyear"], match["rightyear"]]</code> | 在枚举替代行之前，观察的subject/year/unit key必须存在且只有一个数值。缺key或矛盾key就unknown，不能用另一个行救回。 |
| 412 | <code>        # This gate precedes enumeration. Alternative rows never rescue missing keys.</code> | 在枚举替代行之前，观察的subject/year/unit key必须存在且只有一个数值。缺key或矛盾key就unknown，不能用另一个行救回。 |
| 413 | <code>        for condition in observed_years:</code> | 在枚举替代行之前，观察的subject/year/unit key必须存在且只有一个数值。缺key或矛盾key就unknown，不能用另一个行救回。 |
| 414 | <code>            fact = scope["facts"].get((subject, condition, unit))</code> | 在枚举替代行之前，观察的subject/year/unit key必须存在且只有一个数值。缺key或矛盾key就unknown，不能用另一个行救回。 |
| 415 | <code>            if fact is None:</code> | 在枚举替代行之前，观察的subject/year/unit key必须存在且只有一个数值。缺key或矛盾key就unknown，不能用另一个行救回。 |
| 416 | <code>                return roles, [], [], "unknown", "missing_observed_key"</code> | 在枚举替代行之前，观察的subject/year/unit key必须存在且只有一个数值。缺key或矛盾key就unknown，不能用另一个行救回。 |
| 417 | <code>            if len(fact["values"]) != 1:</code> | 在枚举替代行之前，观察的subject/year/unit key必须存在且只有一个数值。缺key或矛盾key就unknown，不能用另一个行救回。 |
| 418 | <code>                return roles, [], [], "unknown", "contradictory_observed_key"</code> | 在枚举替代行之前，观察的subject/year/unit key必须存在且只有一个数值。缺key或矛盾key就unknown，不能用另一个行救回。 |
| 419 | <code>        # Anonymous rows remain parsed source data, but have no literal subject</code> | 匿名行保留为来源数据但不作回答语法的修复见证；候选facts须有命名subject且同单位。 |
| 420 | <code>        # in the declared answer grammar and therefore are not repair witnesses.</code> | 匿名行保留为来源数据但不作回答语法的修复见证；候选facts须有命名subject且同单位。 |
| 421 | <code>        facts = [f for f in scope["facts"].values() if f["unit"] == unit and f["subject"]]</code> | 匿名行保留为来源数据但不作回答语法的修复见证；候选facts须有命名subject且同单位。 |
| 422 | <code>        terms, traces, compatible_seen = [], [], False</code> | 初始化当前scope见证集合；direct枚举每个候选fact的每个数值/出处。 |
| 423 | <code>        if claim["schema"] == "direct":</code> | 初始化当前scope见证集合；direct枚举每个候选fact的每个数值/出处。 |
| 424 | <code>            for fact in facts:</code> | 初始化当前scope见证集合；direct枚举每个候选fact的每个数值/出处。 |
| 425 | <code>                for value, provenance in fact["values"].items():</code> | 初始化当前scope见证集合；direct枚举每个候选fact的每个数值/出处。 |
| 426 | <code>                    term = [role_indices[name] for name, mismatch in (</code> | 若候选行路径或年份不同，分别加入subject/year role，表示使声明匹配此见证要改哪些部分。 |
| 427 | <code>                        ("subject", fact["path"] != subject), ("year", fact["year"] != year)) if mismatch]</code> | 若候选行路径或年份不同，分别加入subject/year role，表示使声明匹配此见证要改哪些部分。 |
| 428 | <code>                    numeric, unmapped = _numeric_repairs("value", amount, value, role_indices)</code> | 加入金额幅度/显式符号修复，累计不可投影符号标志；零修复且无隐式符号问题才是兼容见证。 |
| 429 | <code>                    term += numeric</code> | 加入金额幅度/显式符号修复，累计不可投影符号标志；零修复且无隐式符号问题才是兼容见证。 |
| 430 | <code>                    unprojectable_sign &#124;= unmapped</code> | 加入金额幅度/显式符号修复，累计不可投影符号标志；零修复且无隐式符号问题才是兼容见证。 |
| 431 | <code>                    compatible_seen &#124;= not term and not unmapped</code> | 加入金额幅度/显式符号修复，累计不可投影符号标志；零修复且无隐式符号问题才是兼容见证。 |
| 432 | <code>                    terms.append(term)</code> | 加入金额幅度/显式符号修复，累计不可投影符号标志；零修复且无隐式符号问题才是兼容见证。 |
| 433 | <code>                    traces.append({"path": fact["path"], "year": fact["year"], "value": str(value),</code> | 保存候选路径/年份/Decimal值/单位、repair集合与原表provenance，供审计为何判定不兼容。 |
| 434 | <code>                                   "unit": unit, "repair": term, "source": provenance,</code> | 保存候选路径/年份/Decimal值/单位、repair集合与原表provenance，供审计为何判定不兼容。 |
| 435 | <code>                                   "unprojectable_sign": unmapped})</code> | 保存候选路径/年份/Decimal值/单位、repair集合与原表provenance，供审计为何判定不兼容。 |
| 436 | <code>        else:</code> | binary先按行路径归组，只允许同一subject行的两个年份金额组成程序。 |
| 437 | <code>            by_path = {}</code> | binary先按行路径归组，只允许同一subject行的两个年份金额组成程序。 |
| 438 | <code>            for fact in facts:</code> | binary先按行路径归组，只允许同一subject行的两个年份金额组成程序。 |
| 439 | <code>                by_path.setdefault(fact["path"], []).append(fact)</code> | binary先按行路径归组，只允许同一subject行的两个年份金额组成程序。 |
| 440 | <code>            programs = sum(2 * sum(len(f["values"]) for f in group) ** 2</code> | 估计同路径数值对×两操作符的程序数，超过4096直接unknown，不从截断集合推出排除结论。 |
| 441 | <code>                           for group in by_path.values())</code> | 估计同路径数值对×两操作符的程序数，超过4096直接unknown，不从截断集合推出排除结论。 |
| 442 | <code>            if programs &gt; _MAX_PROGRAMS:</code> | 估计同路径数值对×两操作符的程序数，超过4096直接unknown，不从截断集合推出排除结论。 |
| 443 | <code>                return roles, [], [], "unknown", "witness_enumeration_budget"</code> | 估计同路径数值对×两操作符的程序数，超过4096直接unknown，不从截断集合推出排除结论。 |
| 444 | <code>            for a in facts:</code> | 枚举同路径两facts、各值及加/减，用Decimal计算候选结果；不使用回答标签选择见证。 |
| 445 | <code>                for b in by_path[a["path"]]:</code> | 枚举同路径两facts、各值及加/减，用Decimal计算候选结果；不使用回答标签选择见证。 |
| 446 | <code>                    for av, ap in a["values"].items():</code> | 枚举同路径两facts、各值及加/减，用Decimal计算候选结果；不使用回答标签选择见证。 |
| 447 | <code>                        for bv, bp in b["values"].items():</code> | 枚举同路径两facts、各值及加/减，用Decimal计算候选结果；不使用回答标签选择见证。 |
| 448 | <code>                            for operator in ("+", "-"):</code> | 枚举同路径两facts、各值及加/减，用Decimal计算候选结果；不使用回答标签选择见证。 |
| 449 | <code>                                value = _arithmetic(av, bv, operator)</code> | 枚举同路径两facts、各值及加/减，用Decimal计算候选结果；不使用回答标签选择见证。 |
| 450 | <code>                                term = [role_indices[name] for name, mismatch in (</code> | 候选程序与观察在主语、左右年份、操作符的每个差异，分别加入相应角色。 |
| 451 | <code>                                    ("subject", a["path"] != subject),</code> | 候选程序与观察在主语、左右年份、操作符的每个差异，分别加入相应角色。 |
| 452 | <code>                                    ("left_year", a["year"] != match["leftyear"]),</code> | 候选程序与观察在主语、左右年份、操作符的每个差异，分别加入相应角色。 |
| 453 | <code>                                    ("operator", operator != match["operator"]),</code> | 候选程序与观察在主语、左右年份、操作符的每个差异，分别加入相应角色。 |
| 454 | <code>                                    ("right_year", b["year"] != match["rightyear"])) if mismatch]</code> | 候选程序与观察在主语、左右年份、操作符的每个差异，分别加入相应角色。 |
| 455 | <code>                                program_unmapped = False</code> | 对left/right/result三数再加幅度/符号修复，记录当前程序及整体是否有无法映射的隐式符号变动。 |
| 456 | <code>                                for name, observed, expected in (("left_value", left, av),</code> | 对left/right/result三数再加幅度/符号修复，记录当前程序及整体是否有无法映射的隐式符号变动。 |
| 457 | <code>                                                                 ("right_value", right, bv),</code> | 对left/right/result三数再加幅度/符号修复，记录当前程序及整体是否有无法映射的隐式符号变动。 |
| 458 | <code>                                                                 ("result", result, value)):</code> | 对left/right/result三数再加幅度/符号修复，记录当前程序及整体是否有无法映射的隐式符号变动。 |
| 459 | <code>                                    numeric, unmapped = _numeric_repairs(name, observed, expected, role_indices)</code> | 对left/right/result三数再加幅度/符号修复，记录当前程序及整体是否有无法映射的隐式符号变动。 |
| 460 | <code>                                    term += numeric</code> | 对left/right/result三数再加幅度/符号修复，记录当前程序及整体是否有无法映射的隐式符号变动。 |
| 461 | <code>                                    unprojectable_sign &#124;= unmapped</code> | 对left/right/result三数再加幅度/符号修复，记录当前程序及整体是否有无法映射的隐式符号变动。 |
| 462 | <code>                                    program_unmapped &#124;= unmapped</code> | 对left/right/result三数再加幅度/符号修复，记录当前程序及整体是否有无法映射的隐式符号变动。 |
| 463 | <code>                                if (not term or term == [role_indices["result"]]) and _plausibly_rounded(value, result["value"]):</code> | 只有结果差异且可由显示精度解释时返回unknown，避免把正常舍入判成不兼容。 |
| 464 | <code>                                    return roles, [], [], "unknown", "unresolved_rounding_precision"</code> | 只有结果差异且可由显示精度解释时返回unknown，避免把正常舍入判成不兼容。 |
| 465 | <code>                                terms.append(term)</code> | 收集程序修复项、零修复兼容性与完整左右数值/操作符/结果/来源trace，保留全部允许见证。 |
| 466 | <code>                                compatible_seen &#124;= not term and not program_unmapped</code> | 收集程序修复项、零修复兼容性与完整左右数值/操作符/结果/来源trace，保留全部允许见证。 |
| 467 | <code>                                traces.append({"path": a["path"], "left_year": a["year"],</code> | 收集程序修复项、零修复兼容性与完整左右数值/操作符/结果/来源trace，保留全部允许见证。 |
| 468 | <code>                                               "right_year": b["year"], "left_value": str(av),</code> | 收集程序修复项、零修复兼容性与完整左右数值/操作符/结果/来源trace，保留全部允许见证。 |
| 469 | <code>                                               "right_value": str(bv), "operator": operator,</code> | 收集程序修复项、零修复兼容性与完整左右数值/操作符/结果/来源trace，保留全部允许见证。 |
| 470 | <code>                                               "result": str(value), "unit": unit, "repair": term,</code> | 收集程序修复项、零修复兼容性与完整左右数值/操作符/结果/来源trace，保留全部允许见证。 |
| 471 | <code>                                               "source": [ap, bp], "unprojectable_sign": program_unmapped})</code> | 收集程序修复项、零修复兼容性与完整左右数值/操作符/结果/来源trace，保留全部允许见证。 |
| 472 | <code>        if not terms:</code> | 候选见证族空时不能作封闭排除，返回unknown。 |
| 473 | <code>            return roles, [], [], "unknown", "empty_witness_family"</code> | 候选见证族空时不能作封闭排除，返回unknown。 |
| 474 | <code>        interpretations.append("supported" if compatible_seen else "incompatible")</code> | 当前scope有零修复见证就supported，否则incompatible；同时累积terms及该表原header上下文/spans。 |
| 475 | <code>        all_terms.extend(terms)</code> | 当前scope有零修复见证就supported，否则incompatible；同时累积terms及该表原header上下文/spans。 |
| 476 | <code>        witness_traces.append({"table": scope["table"], "table_span": scope["span"],</code> | 当前scope有零修复见证就supported，否则incompatible；同时累积terms及该表原header上下文/spans。 |
| 477 | <code>                               "header_context": scope["header_context"],</code> | 当前scope有零修复见证就supported，否则incompatible；同时累积terms及该表原header上下文/spans。 |
| 478 | <code>                               "header_context_spans": scope["header_context_spans"],</code> | 当前scope有零修复见证就supported，否则incompatible；同时累积terms及该表原header上下文/spans。 |
| 479 | <code>                               "witnesses": traces})</code> | 当前scope有零修复见证就supported，否则incompatible；同时累积terms及该表原header上下文/spans。 |
| 480 | <code>    if not interpretations:</code> | 没有任何来源锚点解释则unknown。 |
| 481 | <code>        return roles, [], [], "unknown", "no_source_anchor"</code> | 没有任何来源锚点解释则unknown。 |
| 482 | <code>    if "supported" in interpretations:</code> | 全部scope均supported才supported；有supported与incompatible混合则unknown，不武断选择一个表。supported不生成约束因子。 |
| 483 | <code>        status = "supported" if set(interpretations) == {"supported"} else "unknown"</code> | 全部scope均supported才supported；有supported与incompatible混合则unknown，不武断选择一个表。supported不生成约束因子。 |
| 484 | <code>        return roles, [], witness_traces, status, "compatible_witness" if status == "supported" else "mixed_interpretations"</code> | 全部scope均supported才supported；有supported与incompatible混合则unknown，不武断选择一个表。supported不生成约束因子。 |
| 485 | <code>    if unprojectable_sign:</code> | 即使都不兼容，只要修复族包含不能投影的隐式符号改变也弃权，避免强指向不存在的token。 |
| 486 | <code>        return roles, [], witness_traces, "unknown", "unprojectable_implicit_sign"</code> | 即使都不兼容，只要修复族包含不能投影的隐式符号改变也弃权，避免强指向不存在的token。 |
| 487 | <code>    terms = _terms(all_terms)</code> | 合并最小DNF项后超过256就unknown，禁止删项后假称完整事件。 |
| 488 | <code>    if len(terms) &gt; _MAX_TERMS:</code> | 合并最小DNF项后超过256就unknown，禁止删项后假称完整事件。 |
| 489 | <code>        return roles, [], [], "unknown", "repair_term_budget"</code> | 合并最小DNF项后超过256就unknown，禁止删项后假称完整事件。 |
| 490 | <code>    return roles, terms, witness_traces, "incompatible", "closed_incompatible"</code> | 只有封闭、无歧义、可投影且预算内的全部见证均不兼容，才返回incompatible和修复DNF。该结论仅限声明语法/来源表。 |
| 491 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 492 | <code> </code> | 空白行：分隔定义/逻辑段，不执行运算。 |
| 493 | <code>def compile_proofs(source, response, offsets, target, question=None):</code> | compile_proofs是外部入口，输入source/response、原Nx2 offsets、有效target mask或不同整型索引及可选question；返回不兼容DNF组件与中立覆盖诊断。未解析/歧义/支持断言不标任何token为clean。由revision.proof_pack调用。 |
| 494 | <code>    """Return incompatible monotone-DNF factors and neutral coverage diagnostics.</code> | compile_proofs是外部入口，输入source/response、原Nx2 offsets、有效target mask或不同整型索引及可选question；返回不兼容DNF组件与中立覆盖诊断。未解析/歧义/支持断言不标任何token为clean。由revision.proof_pack调用。 |
| 495 | <code> </code> | compile_proofs是外部入口，输入source/response、原Nx2 offsets、有效target mask或不同整型索引及可选question；返回不兼容DNF组件与中立覆盖诊断。未解析/歧义/支持断言不标任何token为clean。由revision.proof_pack调用。 |
| 496 | <code>    ``target`` is a Boolean mask or distinct integer indices into the original</code> | compile_proofs是外部入口，输入source/response、原Nx2 offsets、有效target mask或不同整型索引及可选question；返回不兼容DNF组件与中立覆盖诊断。未解析/歧义/支持断言不标任何token为clean。由revision.proof_pack调用。 |
| 497 | <code>    Nx2 character offsets. No input is mutated or removed. ``question`` is</code> | compile_proofs是外部入口，输入source/response、原Nx2 offsets、有效target mask或不同整型索引及可选question；返回不兼容DNF组件与中立覆盖诊断。未解析/歧义/支持断言不标任何token为clean。由revision.proof_pack调用。 |
| 498 | <code>    retained as optional context; this finite grammar never guesses its intent.</code> | compile_proofs是外部入口，输入source/response、原Nx2 offsets、有效target mask或不同整型索引及可选question；返回不兼容DNF组件与中立覆盖诊断。未解析/歧义/支持断言不标任何token为clean。由revision.proof_pack调用。 |
| 499 | <code>    Missing/ambiguous structures, keys, units or token projections are unknown.</code> | compile_proofs是外部入口，输入source/response、原Nx2 offsets、有效target mask或不同整型索引及可选question；返回不兼容DNF组件与中立覆盖诊断。未解析/歧义/支持断言不标任何token为clean。由revision.proof_pack调用。 |
| 500 | <code>    Supported claims produce no factor and do not label any token as clean.</code> | compile_proofs是外部入口，输入source/response、原Nx2 offsets、有效target mask或不同整型索引及可选question；返回不兼容DNF组件与中立覆盖诊断。未解析/歧义/支持断言不标任何token为clean。由revision.proof_pack调用。 |
| 501 | <code>    """</code> | compile_proofs是外部入口，输入source/response、原Nx2 offsets、有效target mask或不同整型索引及可选question；返回不兼容DNF组件与中立覆盖诊断。未解析/歧义/支持断言不标任何token为clean。由revision.proof_pack调用。 |
| 502 | <code>    if not isinstance(source, str) or not isinstance(response, str):</code> | 来源和回答必须为字符串，拒绝其他对象默默转文本。 |
| 503 | <code>        raise TypeError("source and response must be strings")</code> | 来源和回答必须为字符串，拒绝其他对象默默转文本。 |
| 504 | <code>    offsets = np.asarray(offsets)</code> | offset与target转数组，后续不原地修改。 |
| 505 | <code>    target = np.asarray(target)</code> | offset与target转数组，后续不原地修改。 |
| 506 | <code>    if offsets.ndim != 2 or offsets.shape[1] != 2 or not np.issubdtype(offsets.dtype, np.integer):</code> | offset必须为整数[N,2]；不能将小数位置截断当合法字符坐标。 |
| 507 | <code>        raise ValueError("offsets must be an integer Nx2 array")</code> | offset必须为整数[N,2]；不能将小数位置截断当合法字符坐标。 |
| 508 | <code>    if target.ndim != 1:</code> | target只允许一维mask或索引。 |
| 509 | <code>        raise ValueError("target must be a one-dimensional mask or index array")</code> | target只允许一维mask或索引。 |
| 510 | <code>    if target.dtype == bool:</code> | bool mask长度必须等于原offset行数，flatnonzero转换为原token索引。 |
| 511 | <code>        if len(target) != len(offsets):</code> | bool mask长度必须等于原offset行数，flatnonzero转换为原token索引。 |
| 512 | <code>            raise ValueError("target mask must match offsets")</code> | bool mask长度必须等于原offset行数，flatnonzero转换为原token索引。 |
| 513 | <code>        selected = np.flatnonzero(target)</code> | bool mask长度必须等于原offset行数，flatnonzero转换为原token索引。 |
| 514 | <code>    elif np.issubdtype(target.dtype, np.integer):</code> | 整数target保留给定顺序，仅统一到int64，不擅自排序。 |
| 515 | <code>        selected = target.astype(np.int64, copy=False)</code> | 整数target保留给定顺序，仅统一到int64，不擅自排序。 |
| 516 | <code>    elif target.size == 0:</code> | 空target允许转换为空整数索引；非空浮点/其他类型拒绝。 |
| 517 | <code>        selected = np.empty(0, dtype=np.int64)</code> | 空target允许转换为空整数索引；非空浮点/其他类型拒绝。 |
| 518 | <code>    else:</code> | 空target允许转换为空整数索引；非空浮点/其他类型拒绝。 |
| 519 | <code>        raise ValueError("target indices must be integers")</code> | 空target允许转换为空整数索引；非空浮点/其他类型拒绝。 |
| 520 | <code>    if (np.any(selected &lt; 0) or np.any(selected &gt;= len(offsets))</code> | target索引必须不重复且在[0,N)内，避免同token多次进入事件。 |
| 521 | <code>            or len(np.unique(selected)) != len(selected)):</code> | target索引必须不重复且在[0,N)内，避免同token多次进入事件。 |
| 522 | <code>        raise ValueError("target indices must be distinct and in range")</code> | target索引必须不重复且在[0,N)内，避免同token多次进入事件。 |
| 523 | <code>    if (np.any(offsets &lt; 0) or np.any(offsets[:, 1] &lt; offsets[:, 0])</code> | 每offset字符范围非负、end≥start且不越response长度；不额外假定所有token offsets互不重叠。 |
| 524 | <code>            or np.any(offsets &gt; len(response))):</code> | 每offset字符范围非负、end≥start且不越response长度；不额外假定所有token offsets互不重叠。 |
| 525 | <code>        raise ValueError("offsets must be valid response character intervals")</code> | 每offset字符范围非负、end≥start且不越response长度；不额外假定所有token offsets互不重叠。 |
| 526 | <code>    scopes, source_problems = _source_scopes(source)</code> | 解析来源scope，准备结果，找有限声明并检查整个回答作用域；无模型或标签调用。 |
| 527 | <code>    components, diagnostics = [], []</code> | 解析来源scope，准备结果，找有限声明并检查整个回答作用域；无模型或标签调用。 |
| 528 | <code>    claims = _claims(response, scopes)</code> | 解析来源scope，准备结果，找有限声明并检查整个回答作用域；无模型或标签调用。 |
| 529 | <code>    assertion_scope = _assertion_scope(response, claims)</code> | 解析来源scope，准备结果，找有限声明并检查整个回答作用域；无模型或标签调用。 |
| 530 | <code>    for claim in claims:</code> | 为每claim附同一个整回答作用域结果，再用封闭见证规则评估。 |
| 531 | <code>        claim["assertion_scope"] = assertion_scope</code> | 为每claim附同一个整回答作用域结果，再用封闭见证规则评估。 |
| 532 | <code>        roles, terms, witnesses, status, reason = _evaluate(claim, scopes)</code> | 为每claim附同一个整回答作用域结果，再用封闭见证规则评估。 |
| 533 | <code>        tokens = _role_tokens(roles, response, offsets, selected)</code> | 将角色spans严格投影到selected token；投影歧义覆盖先前状态为unknown，不能继续施加proof因子。 |
| 534 | <code>        if tokens is None:</code> | 将角色spans严格投影到selected token；投影歧义覆盖先前状态为unknown，不能继续施加proof因子。 |
| 535 | <code>            status, reason = "unknown", "ambiguous_token_projection"</code> | 将角色spans严格投影到selected token；投影歧义覆盖先前状态为unknown，不能继续施加proof因子。 |
| 536 | <code>        record = {"schema": claim["schema"], "span": list(claim["span"]),</code> | 所有声明都记录schema、字符范围、状态/原因、角色和来源见证，包含未生成因子的neutral诊断。 |
| 537 | <code>                  "status": status, "reason": reason, "roles": roles,</code> | 所有声明都记录schema、字符范围、状态/原因、角色和来源见证，包含未生成因子的neutral诊断。 |
| 538 | <code>                  "source_interpretations": witnesses}</code> | 所有声明都记录schema、字符范围、状态/原因、角色和来源见证，包含未生成因子的neutral诊断。 |
| 539 | <code>        diagnostics.append(record)</code> | 所有声明都记录schema、字符范围、状态/原因、角色和来源见证，包含未生成因子的neutral诊断。 |
| 540 | <code>        if status == "incompatible":</code> | 只有incompatible生成role_tokens+repair_terms组件，并附provenance；不把supported当负标签输入模型。 |
| 541 | <code>            components.append({"role_tokens": tokens, "repair_terms": terms,</code> | 只有incompatible生成role_tokens+repair_terms组件，并附provenance；不把supported当负标签输入模型。 |
| 542 | <code>                               "status": status, "provenance": record})</code> | 只有incompatible生成role_tokens+repair_terms组件，并附provenance；不把supported当负标签输入模型。 |
| 543 | <code>    if not diagnostics:</code> | 一个声明都没识别时明确记录no_supported_claim_schema；这里supported_claim_schema指被实现的语法，不是内容被来源支持。 |
| 544 | <code>        diagnostics.append({"schema": None, "status": "unknown", "reason": "no_supported_claim_schema"})</code> | 一个声明都没识别时明确记录no_supported_claim_schema；这里supported_claim_schema指被实现的语法，不是内容被来源支持。 |
| 545 | <code>    counts = Counter(record["status"] for record in diagnostics)</code> | 统计各状态和具体弃权原因，便于看有限语法覆盖而非只报成功例子。 |
| 546 | <code>    reasons = Counter(record["reason"] for record in diagnostics)</code> | 统计各状态和具体弃权原因，便于看有限语法覆盖而非只报成功例子。 |
| 547 | <code>    return {"components": components, "diagnostics": {</code> | 返回components及版本explicit_json_year_table_v2、所有claim诊断、三状态计数和source结构问题。 |
| 548 | <code>        "schema": "explicit_json_year_table_v2", "claims": diagnostics,</code> | 返回components及版本explicit_json_year_table_v2、所有claim诊断、三状态计数和source结构问题。 |
| 549 | <code>        "counts": {status: counts[status] for status in ("supported", "incompatible", "unknown")},</code> | 返回components及版本explicit_json_year_table_v2、所有claim诊断、三状态计数和source结构问题。 |
| 550 | <code>        "reasons": dict(reasons), "source_problems": source_problems,</code> | 返回components及版本explicit_json_year_table_v2、所有claim诊断、三状态计数和source结构问题。 |
| 551 | <code>        "source_tables": len(scopes), "source_facts": sum(len(s["facts"]) for s in scopes),</code> | 报告表/事实key数量、原始/有效token数以及question是否提供；question没有参与意图猜测或作为外部事实证明。 |
| 552 | <code>        "original_token_count": len(offsets), "scored_token_count": len(selected),</code> | 报告表/事实key数量、原始/有效token数以及question是否提供；question没有参与意图猜测或作为外部事实证明。 |
| 553 | <code>        "question_present": question is not None,</code> | 报告表/事实key数量、原始/有效token数以及question是否提供；question没有参与意图猜测或作为外部事实证明。 |
| 554 | <code>    }}</code> | 报告表/事实key数量、原始/有效token数以及question是否提供；question没有参与意图猜测或作为外部事实证明。 |
