# H020紧凑固定schema、anchor义务与确定性验证合同

这是未来实现合同，不是已实现parser。以下schema说明全部请求相同；`|`表示enum备选，实际输出只取一个。所有keys准确、无额外/duplicate JSON keys、finite JSON。arrays紧凑以控制1536输出预算，必须保留列出的语义字段。

```json
{
  "records": [
    {
      "id": "r1",
      "status": "complete|partial|unknown",
      "speech": "asserted|instruction|reported|hypothetical|unknown",
      "anchors": ["p1a0"],
      "cite": [[1, "exact passage substring", 0]],
      "pred": ["exact predicate substring", 0],
      "args": [["subject|object|value|recipient|location|other", "exact substring", 0]],
      "ops": [["negation|quantifier|condition|time|unit|rate|comparison|modality|quotation|attribution|sequence|alternative|method|other", "exact substring", 0, "record|unknown"]],
      "links": [["before|after|condition|reported_by|same_event|alternative|coreference|unknown", "r2", 0]],
      "unknown": ["unresolved issue"]
    }
  ],
  "coverage": [
    {
      "passage": 1,
      "unknown": [["reason", ["p1a1"]]],
      "nonrelational": [["reason", ["p1a2"]]],
      "scope_unknown": [["exact passage substring", 0, "reason"]]
    }
  ]
}
```

cite item=`[passage,quote,zero-based exact occurrence]`；pred=`[text,cite_index]`或null；args=`[role,text,cite_index]`；ops=`[kind,text,cite_index,scope]`；links=`[kind,other_record_id,cite_index]`。三个原passage均恰有一coverage项，按原passage identity而非呈现位置。id唯一；link引用须存在，但validator不自动证明语义、不做logic closure/去环。所有arrays允许空但不能少key。

模型wire在首fencedJSON之前固定加：`Vertical bars denote alternatives; output exactly one option, never the pipe-separated literal. All keys are required; arrays may be empty when unused. Example entries illustrate shapes, not source facts; every reference must point to an existing entry.` 然后接首fencedJSON与本段之前的array legend段。不能只发含pipe的示例而漏掉这一规则。

所有role/operator的text默认非空exact原文；只有status≠complete且unknown有对应义务时允许null。predicate允许null但不可complete。没有明确unit的无量纲计数不能被补造计量单位；需有具体被计数对象。任何无法用record级ops表示的嵌套negation/quantifier/conditional scope应unknown，不靠拆records丢掉operator。other不是未审语义的自动许可。

## M0 source-only lexical anchor分母（不以R选择）

只扫描passage body，不扫描question/header/footer。规则逐项独立、允许不同类别重叠但物理identity保留；按passage ID、start、end、kind、literal排序给`p{ID}a{ordinal}`。一切原数字均保留，不因预判是步骤编号、URL、年份或单位不匹配而丢掉。

1. `number`：每个 `\d+(?:[.,]\d+)*` exact match；负号/范围/分数不在此阶段解释，`5 1/2`保留各digit occurrence，原完整数字表达必须在record的value/operator中保留。
2. `number_word`：case-insensitive Unicode word boundaries，固定词表`zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty hundred thousand million billion half quarter third once twice thrice`。
3. `unit`：完整word，固定`second(s) sec(s) minute(s) min(s) hour(s) hr(s) day(s) week(s) month(s) year(s) degree(s) pound(s) lb(s) percent percentage dollar(s) cent(s) mile(s) meter(s) metre(s) kilometer(s) kilometre(s) inch inches foot feet kg g mg cm mm`。括号表示列出的singular/plural变体，不是原文括号。大小写只用于检测，输出literal保持原样。
4. `scope_cue`：完整word，固定`each every all both either neither none no any some most least more less fewer only always never exactly about another additional per between during before after afterwards then when until if unless not without`。它们不是已解析operator，相关scope未解应unknown；非量化上下文也不静默删除。

输入anchor wire固定四列`[id,kind,literal,[start,end]]`，id的`p1/p2/p3`编码原passage ID，span是body-local Python字符半开区间。不含owner/predicate/真假。每anchor必须在**records.anchors之并集**、coverage.unknown或coverage.nonrelational恰属一类。一个mapped anchor可属于多个records（重复使用/多对多合法）；不能因这种重用报重复错误。unknown/nonrelational中不得重复/交叉，不能与mapped重叠。records的每anchor必须存在且被其citation覆盖；非关系步骤编号/URL分类仍需独立S-only审查，不能让模型自己宣布无关系而免义务。审计同时给raw-mapped和valid-citation-covered账本：坏record/坏cite不得假计valid-mapped，原模型自述完整保留但不能填补覆盖门。

模型不必复述全部anchor literal，仅引用ID；deterministic checker重建coverage分母，另以独立S-only ledger审重要的非anchor条件/指代/单位缺口。全部anchors映射不证明语义全覆盖。

## Exact citation/角色边界

passage∈{1,2,3}、occurrence为非bool非负整数、quote非空。枚举原body所有exact、允许overlap匹配，选择ordinal；无匹配/越界拒绝record，不改ordinal、不选最近、不normalize空格或标点。计算原prompt/body-local字符spans并保存SHA。不同记录/相同值/重复证据不按字符串去重；物理citation重复不增加独立证据量，保留并标结构重复。

每个role text在指定cite内的所有literal matches全部保存；多匹配而schema未进一步定位时为boundary ambiguity unknown，不取first/nearest。跨passage不能伪造单quote；需多cite。该规则避免要求模型心算绝对字符坐标，但不假装解决scope或coreference。

parser只接普通JSON，不剥fence、不截取花括号、不修JSON/空格/原文。max_new_tokens或invalid envelope保留raw/tokenIDs并标request unknown，不salvage局部records。valid envelope中的bad record独立保留/reject；任何原自述complete的语义错误使自动可用门失败，不由人工删除后晋升。

机械边界固定：invalid JSON、duplicate JSON keys、nonfinite值或错误top-level containers令整request envelope unknown。合法envelope的逐record问题完整保留；所有重复record id的records全部拒绝，不挑有利一条，指向该id的links标ambiguous。partial/unknown的null-role仅机械要求unknown非空，其语义是否对应null字段留给外审，不假造自动scope证明。

## 研究许可与unknown

模型status永远是proposal。只有complete、pred非null、至少一个explicit argument/value、unknown为空、无scope=unknown/null/边界歧义，且independent S-only review完整通过，才可另给研究许可。字面/JSON通过不等于source entailment/事实对错，coverage全mapped也不证明没有省掉非anchor条件。

研究边界：四S/controls是已暴露exploratory材料；actualmessages可证明extractor本次R-blind，不把人工reviewer既往R exposure隐去或称双盲。source-only ledger不是部署真值或目标hallucination标签。NF4/repetition1.05/1536导致的引用/JSON/遗漏失败只限制该instrument/prompt/预算，不扩成LLM普遍不能理解来源。
