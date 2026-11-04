# H020实验计划：来源信息条件资格

**问题**：在不看R时自动产生完整量化关系候选，并明确条件/引用/unknown。
**方法命题**：先检查新source-only语义信息条件是否可用，再谈prefix-L/内部repair；本轮不承诺检测方法。
**日期**：2026-11-05。正式合同以同版本PROPOSAL/PROTOCOL/PROMPT/SCHEMA为准，等待独立review。

## Claim map

| 主张 | 最低证据 | Block |
|---|---|---|
| C1 有限source-only关系提议可用 | 全4原S各≥1完整非NER量化关系；全部anchors/义务归账；无wrong-complete；两个controls必要关系通过 | B1–B3 |

反主张：引用/JSON正确即语义正确、只抽易事实/隐去条件、人工删错后自动晋升、按R/错误选facts。

## 紧凑实验故事

### B1 — source identity/控制/义务资格（MUST）

- 4原S＋14375两个source-only控制；0LM。固定数字/word-number/unit/scope-cue anchors，不看R。
- 独立S-only obligation ledger和literal control许可先冻结。所有输入/变换/prompt/schema/模型文件及token预算身份封存。
- 单输入超过4096拒绝，无截断/重试。任何控制越权0generation退出。

### B2 — 固定Qwen3B提议（MUST）

- 6请求最大；同generic prompt/schema、NF4/FP16/SDPA，1536output上限。
- 资源实际postimport/preload/prerequest严门；最多9216actualdecoderforwards，逐pre/posthook计费，不是6forward。
- 全部raw/tokenIDs/effective generation config/失败/资源保存；无R答案、训练、额外canary、prompt sweep或截断救援。

### B3 — 全输出独立S-only审查（MUST）

- 引用/角色/条件/operator/speech/完整性分别审查；数字分母和unknown不丢。
- reorder局部facts/provenance及unknown不变；swap仅两localduration跟原文变，重复/其它facts保留。
- 全门通过也仅可提下一项prefix-L零LM资格；手工筛选后的可用子集不是自动前端通过。

### 刻意不做

- H019 χ/MI风险、head/layer selection、pair幅度救援；新8B/7B下载/量化；R-first18次重演。
- 原R重生成、新问答/反事实答案、repair/prefix-L执行、风险/AUROC、GT。
- H010与四QA不同source覆盖率不相减，frontier necessity尚未由本小实验建立。

## 运行顺序

| M | 固定任务 | 模型预算 | 决策 |
|---|---|---|---|
| M0 | 输入/control/anchors/source义务/token预算 | 0 | 通过才审实现/新claim |
| M1 | 假backend/installed identity/actualpostimport资源 | 0LM | code+resource failclosed |
| M2 | 6个预定S-only提议 | ≤6requests/9216decoderforwards | 格式/语义失败保留，不改请求；资源/工程失败封存停 |
| M3 | 独立S-only全量语义/controls审查 | 0 | C1失败关闭当前自动front；通过仅下一零LM提案 |

## 成本与局限

- 0训练、模型资产免下载。总input≤24576/output≤9216 tokens；实际GPU时间/峰值未测，不预报达成。
- 输出1536仍可能不够；范围已限定量化库存但完整条件不能省。截断就是unknown，无追加calls。
- 最大风险是quoted词都在而关系/scope仍错；独立审查所有自述complete，不给模型语义投票当真值。
- 本四source既有暴露，不宣称新确认集/论文效果。尚无完整无监督检测结果。

## 当前检查

- [x] 一个主张、两个固定source controls、少量实际请求。
- [x] 只改变S-only信息条件，明确区别H002/H008/H010。
- [x] 原始数字coverage/unknown、真正decoder成本、fixed effective decode已规定。
- [ ] 独立科学review通过。
- [ ] 实现/代码审查/installed identity/新claim/actual资源资格。
- [ ] 提议输出与独立语义资格。
