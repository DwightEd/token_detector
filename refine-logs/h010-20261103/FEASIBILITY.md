# H010 自然 named / quantity span 前端：工程范围与后续接口

本轮只准备冻结的语言学前端与 inference-only span inventory。没有运行 QA、检测分数、标签评价或参数搜索；span 数量不能说明语义可回答率、事实支持率或 detector coverage。这也不是旧 finite-certificate 规则覆盖的重跑。

## 官方身份与适用范围

- 固定 spaCy **3.8.18**；官方 PyPI 声明 Python `>=3.10,<3.16`，提供 CPython 3.13 Windows x64 wheel，MIT。[PyPI 固定版本](https://pypi.org/project/spacy/3.8.18/)
- 固定 **en_core_web_sm 3.8.0**，官方发行 wheel 12,806,118 bytes；模型要求 spaCy `>=3.8.0,<3.9.0`，MIT。该发行资产没有公布 GitHub digest，故明确保留“官方 HTTPS + 官方大小核对 + 本地流式 SHA256 冻结”的身份边界；不把本地 hash 冒称上游公布 hash。[官方发行](https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.8.0)、[官方模型元数据](https://github.com/explosion/spacy-models/blob/master/meta/en_core_web_sm-3.8.0.json)
- 模型是 CPU-oriented、无静态词向量的英文小管线。训练来源包括 OntoNotes 5；NER、POS、dependency parser 均为**外部监督**的冻结工具，不是 target benchmark gold 上的训练，也不声称无数据污染。[官方模型说明](https://spacy.io/models/en)
- 原生实体类型包括 PERSON、ORG、GPE、LOC、DATE、TIME、MONEY、PERCENT、QUANTITY、CARDINAL 等；不添加自定义实体规则、EntityRuler 或 task-specific 手工关系模板。输出使用原生 `start_char` / `end_char`，Python Unicode code point 的半开区间。[NER 文档](https://spacy.io/usage/linguistic-features#named-entities)、[Span API](https://spacy.io/api/span)

## 本轮冻结操作

独立环境 `D:/projects/research/envs/token-detector-nlp`；不修改 repo `.venv`、GPU 环境或既有模型。仅 CPU、单进程、线程上限 2；启用 `tok2vec/tagger/parser/attribute_ruler/ner`，禁用 lemmatizer。每字段预算 100,000 characters / 20,000 tokenizer tokens；超预算保留原 case，明确 unknown，不截断、不删例、不静默分窗。此预算是工程限制，不是按任务真值调参。

输入只打开 `outputs/research-goal-20261102/case_panel_v1/inputs.jsonl`，字段白名单 `id/source/response`，原 18 条全部保留。不会打开邻近 manifest、evaluation_only、gold 或既有 prediction/score 文件。source 与 R 各自独立做语言学处理；两侧实体相同、相异、缺失均不被解释成支持或矛盾。

先做 5 个合成工程 probe，包含重复姓名与金额、emoji 前缀、重音与组合附加符号、换行/制表/尾空白、空串。仅验证字符串与 offset 契约，不评估 NER 语义准确率。天然 NER 的漏检、类型错误和别名分裂仍可能存在；保留每一个**已输出** occurrence 并不意味着找到原文所有实体。

库存对每字段保存：文本 SHA、长度、预算状态、原生实体 type 与 exact span/text、重复 occurrence ID、token POS/tag/dependency/head，以及句子边界。没有 source entailment、语义真值或 QA 答案。

## 后续 cloze 接口建议（本轮不实现）

建议先冻结一个可审计的 proposal，而非直接接 scorer：

```
propose_cloze(source_inventory, response_inventory,
              target_occurrence_id, masking_policy_identity)
  -> {status, reason, source_identity, response_identity,
      target_span, target_type, masked_response,
      mask_spans, protected_non_target_context, candidate_inventory_identity}
```

`target_span` 必须来自已冻结 R inventory，并与原串 exact slice 相等。masked_response 应同时遮蔽目标原槽值的所有**字面 occurrence**（即便某次 NER 未标到）及 R 中所有同类型其他值的 occurrence，合并相交区间、从右到左替换，并逐项保存 original span -> placeholder 映射。这一步使用 R 值只用于擦除，原值及替代值不得再拼回 query/prefix。重复槽值必须全部擦除，不按唯一字符串去重后丢位置。

同类型的最保守初版采用精确原生 NER label；PERSON↔ORG、GPE↔LOC、CARDINAL↔DATE 等类型不一致、嵌套/重叠、别名、未知类型，以及疑似残留数字/实体时都应显式 unresolved，不能把未检测到的别名安全性视为已经验证。未经单独前瞻登记，不扩展为经验性的 label 合并或关系启发式。other-type 的上下文值可能是必要关系锚点，也可能泄漏答案；proposal 必须列出它们，后续接口审查才能决定允许策略。本轮不宣称该审查已经通过。

候选值应从 **S 独立解析**的同类型 inventory 产生，先冻结候选清单/排序与去重映射；R 的 target value、原槽值匹配、频率和任何 gold 均不得用来保留、删除或优先候选。S 中包含候选原值是正常证据，不能误当 response-value leak。候选不全、无同类型值、跨度超预算都保留 unknown；source 文本有匹配字符串不等于问题可答。

任何后续 QA 工程须另行冻结：怎样从被擦除的 R 生成 query、如何处理跨句指代/句法破坏与数值单位、源候选可变长度/EOS 的模型接口，以及 unknown 保留方式。本轮不写 QA 或评分，也不据本库存的类型/count 选择最优方法。

## 可复核文件

`readiness.json` 与 `requirements-locked.txt` 固定环境、每个 wheel 的 SHA256、许可、来源和磁盘门；`pilot-freeze.json` 在 probe / 自然 NLP 处理之前固定输入/脚本/模型文件身份与操作预算。`synthetic-probe.json`、`pilot-inventory.jsonl`、`pilot-summary.json` 和 `pilot-seal.json` 保留实际工程观察、unknown、耗时和哈希。
