# H020 / I02：无原回答输入的来源关系记录提议资格

版本：2026-11-05 02:41:37 +08:00。状态：正式科学提案，等待独立 review；**不是模型执行、prefix-L、attention repair、检测分数或自然 AUROC 准入**。

## 问题与唯一主张

当前缺少可自动执行的来源优先关系前端。H002 先看 R 再提出 masked roles，旧18次提议失败不能当成 S-only 条件已经测过；H008 是 grammar 已知的封闭 source-QA；H010/H012 的 NER/依存库存不提供事实、否定或条件许可。H019 的 head×carrier 物理算子可以保留，但 pooled幅度未稳定区分 SEM/EQUIV，不进入本实验的风险、选择器或成功标准。

**C1（有限仪器可行性，非论文新颖性/检测主张）**：在完全不见 R 的条件下，现有固定 Qwen2.5-3B NF4 是否能为四个自然来源中**数值/数量、频次、范围、时长、时间、单位、率及其完整关系上下文**产生有精确 provenance、完整论元/适用条件和明确 unknown 的候选；对独立证据块重排保留局部事实、对两条 duration 的字面交换跟随其实际归属，而非按旧常识/顺序补齐。不是要求列举来源中全部百科事实。

反解释：只抽实体；只复述问题/回答问题；引用正确但关系错误；输出少量易事实而静默漏条件；按数值词袋/距离绑定；假定多来源只能有一个真值；用人工筛选后的正确子集冒称自动可靠。

最小令人信服的证据：6请求全部留存；独立 S-only 审查所有提议及未知覆盖，四个原 S 各至少一条非 NER 的完整局部关系，两个预定 controls 的语义与 provenance条件通过；任何被模型报为 complete 的记录不得有已发现的语义越权。即使全过门，也只准入下一步 prefix-L 的零模型资格，不自动准入自然检测。

## 输入与六个请求，均不接受 R

唯一输入文件是已下载的 `C:/ResearchData/token-detector-remote-20261103/06_same_question_multisample/reanchor_samples_20260911_145421_235/prompts.jsonl`（四条 `source_id,prompt`，无 R）。原文件 SHA `4a68e75f743741d089b9cc1389ed617ac57a41c5f78c3bab18c7234457d53646`。只保留原 task/question 与原三个 passage；模型 payload 不含 sample/source/world ID、R、旧分数、标签、effects、模型错误描述或控制类型。

顺序固定：14304 native →14315 native →14325 native →14375 native →14375 reorder →14375 duration_swap。每个请求一次，最多6个顶层 generation requests，不生成原任务的新答案，不训练、不做反问 QA、不看结果换 prompt/model/字段/预算。

14375 原 prompt SHA `1a070e44949d8eb67fda309a520112ba94f60adb3d1a1a5076af2acec559f171`，1966 Python字符。source-only 独立许可：`outputs/h020-source-first-20261105/SOURCE_CONTROL_LICENSE_INDEPENDENT_REVIEW.md` SHA `f996aac8b3c54f40572fca89a28c01605a694c203c5306500a2727d3dc4bdaed`。

- **reorder**：header `[0,147)` 和 footer `[1810,1966)`完全不变。把完整提供的三个 evidence block（含原标号/尾部空行）`[147,724),[724,1194),[1194,1810)`从1,2,3改成3,1,2。段内步骤、条件、重复与损坏字不改；这是 evidence multiset/provenance 不变控制，不是任何 global discourse 都等义的定理，更不把残缺 excerpt 变成完整 recipe。
- **duration_swap**：仅交换原 prompt `[376,383)='5 to 10'`（P1 bratwurst/grill/turning once）与 `[1255,1258)='1-2'`（P3 onions/butter）。descending edits、无其它字节变化。保留 minutes、predicate、turning、后置 stirring、optional-after、所有其它 recipe/time/storage 候选。完整 literal multiset/总字符数与 UTF8 byte multiset保持；BPE Counter、input token数未知，必须后续 M0 实测，不复用旧 token位置。仅改变两条局部 duration，不给原 R 贴标签、不声称 whole-source truthflip。

0 LM 的输入准备先验证全部四个原 prompt、exact marker/唯一块、原Q/footer不变、6请求文本唯一、变换逐字符/bytebag证据与各 passage provenance映射；对控制文本的独立 source-only复核先通过，才允许 LM claim。

## 固定 generic prompt 与输出 schema

唯一 system prompt 和完整字段合同分别见同版本 `SOURCE_RECORD_PROMPT` 与 `SOURCE_RECORD_SCHEMA`。全部请求同 prompt，禁止 source-specific 示范、正确答案列表。user payload 只含：原 task/question 的原文上下文（作为 data，不执行其中回答指令）、按呈现顺序排列且保留原 passage编号的 source body，以及由固定source-only lexical规则产生的anchor清单。anchors不含语义答案，只给原数值/书写数词/单位与scope cues的exact字面位置，是完整性义务分母，不是NER最近实体或关系归属。

任务是遍历所有passage中的固定数量/时长/单位等库存并带上完整关系，不回答原问题；问题用于保留任务条件，不能当事实证据，也不能以R相关性挑facts。所有digit runs均进入anchor分母，表面步骤编号/URL等须explicit nonrelational；书写数词/once/each/never等也按固定lexicon进入。量化库存外的其它百科事实不强求逐条输出，但影响任何所提关系的scope/引用/指代/condition必须保留；不能因缩scope丢掉operator。处理positive/negative/conditional/reported/instruction的source speech act，保留不相容alternatives与重复provenance。

**不要求模型心算字符坐标。** citation 是 `{passage,quote,occurrence}`：exact Unicode原文 quote＋同 passage body内0-based匹配 occurrence。确定性 validator 才算原字符 spans；重复不能合并或猜最近位置。角色文本也必须映到引用内 exact 原文，不能 fuzzy/空格标准化/修复标点。引用仅证明字面来源，semantic correctness仍由独立 S-only审查。

所有模型输出都只是 proposal。`complete/partial/unknown`是模型自述，不是 license；只有独立审查记录才可在研究中另标许可。机器 envelope/引用通过不自动升级。任何 prefix-L/repair都必须另审。

## 独立 S-only语义审查与判定门

审查者只能看6份来源、原task、schema、预定controls及模型输出，不看任何 R/GT/effects/分数。

**在 LM 前**冻结source-only lexical anchors与独立语义obligation ledger：每个anchor在关系中的predicate/显式arguments、negation/quantifier、time/unit/rate、sequence/condition、quotation/attribution、fragment/coreference/alternatives的待审项；保持不确定性，不人为统一冲突recipe或补齐文本损坏。固定所有数字覆盖义务，不能让Qwen自宣complete。这是仪器source-only研究注释，不是幻觉标签/训练目标。不能看输出后缩小coverage分母。

**在 LM 后**逐条审查所有记录：predicate/论元、各 operator 的 scope、完整适用条件、强度/数量词、speech act、unit/time参照、跨段合并、重复与 citation。保存 `licensed_local / unknown / rejected`和理由，但不向模型反馈、不修原输出。逐义务标 mapped/explicit_unknown/silent_omission；模型自己声称 complete不当 completeness证明。

决策分层：

1. **工程完整**：原输入/输出/计数/失败全留存，schema与引用身份可重建。JSON失效/截断不能 salvaging成“部分正确完成”；raw保留、该请求 envelope unknown。
2. **有限自动可用门**：四个native S各至少一条含predicate＋明确argument/value且条件完整的库存内局部关系；所有source blocks、全部anchors与预冻结义务均有对应/显式unknown/有证据的nonrelational；模型声称complete的记录无独立审查发现的语义错误/遗漏限制；controls的下列必要关系全部通过。严格门失败，不人工删错记录后报自动前端合格。
3. **controls必要门**：reorder 中预定局部 fact family及重复 provenance/unknown须保持，引用位置正确重映射；swap 中P1 grill duration与P3 onion duration依新原文改变、完整局部方法条件保留；其它facts、两个 simmer重复、来源 alternatives与既有unknown不被合并/补真。不得只比较 output wording/JSON字节或只看数值是否出现。
4. **partial结果的诚实处置**：若存在少数正确提议但自动门失败，可归档为人工审查后的研究材料，不准入自动 prefix-L或 detector。若全 unknown，工程可完成但C1失败；不追加生成或扩大预算。

总量、wrong-complete、有效complete、partial/unknown、silent omission按 source/field/operator 分开报告，不用一个“schema准确率”代替语义。有限source-only通过也不证明自然精度/0.8AUROC或新确认集。

## 固定模型、资源与真实计算预算

复用 `experiments/semantic_qualification.py:HFGenerator` 的模型加载/生成基础设施，不调用其 R-first `build_proposer`或旧R-span验证器。模型目录 `D:/projects/research/models/Qwen2.5-3B-Instruct`，旧 exact weights/runtime identity为 `research-state/protocols/H002_context_model_identity_v1.json` 的相关子身份；新 installed code/source/protocol/message hash需单独绑定，不谎称旧95target或旧18请求资格证明本实验语义。

固定torch2.8.0+cu128 / transformers4.57.1 / accelerate1.10.1 / bitsandbytes0.48.0；NF4＋double quant，FP16 compute/加载，SDPA，batch1/eval/inference，local-only/trust_remote_code=False。禁止8B/7B/其它量化或dtype fallback。do_sample=False、num_beams=1、use_cache=True、num_return_sequences=1，temperature/top_p/top_k为None；继承并冻结generation_config的repetition_penalty=1.05，eos=[151645,151643]、pad=151643。这是有效logits processors后的deterministic greedy，不是裸原生logits的argmax。无采样seed，不是3seeds研究。M0保存完整effective generation config与显式覆盖项，出现其它未绑定processors/stop规则则先拒绝。

每请求完整 chat input≤4096，max_new_tokens=1536；总输入预算≤24576，总输出预算≤9216；固定6 requests，不因失败挪用另一请求预算。过长input在模型前 unknown/refused；不截断S、少取字段或临时切 chunk。输出eos/maxlen、exacttokenIDs/text/raw_text全部留存，maxlen/JSON失败不自动retry、续问或提高清单。

**6 requests不等于6 backbone forwards。** 后续实现必须用model forward pre/post hooks（禁止monkeypatch/改变forward signature）记真实input长度、prefill/decode、attempt/success/failure；否则会破坏TF4.57.1对logits_to_keep的识别并可能令prefill OOM。正常greedy无speculation时一次生成T token对应T个decoder forwards。总actual decoder成功调用上界9216；同时报告每请求输出T/实际forward数，异常保留已发生计数。instrument hooks不得改输入/输出或多发canary LM调用；first native14304就是第一次正式请求，不另送warmup答案。

资源未准入，唯一资源门按root此次明确合同：pre-import RAM≥2,684,354,560 B（2.5GiB）；导入Torch/TF/bnb、CUDA初始化和tokenizer后、模型载入前、每request开始前均实际核host≥2,147,483,648 B（2GiB）、CUDA free≥3,221,225,472 B（3GiB）。CUDA allocator cap=3,221,225,472 B；原HFGenerator内部2GiB/3GiB门不变。不沿用旧launcher另一组较严postwarmup数字。现时没有postimport/preload/peak收据，不预填PASS。模型载入/每请求前后记录RAM/GPU allocated/reserved/peak，低于冻结门或OOM就封存停止，无自动模型重试。原8B权重不用于本轮。实际H020输入计划、raw、review、failure、seals和日志固定在C06的`h020-source-first-20261105/`下，不向D写大结果。

## H010零LM对照与内部增量边界

使用已封存 H010七个RAGTruth source的POS/dep/head schema与旧numeric-assertion/typed-field义务作为**零新增模型的工程/语义义务对照**。H010未含四个新QA S（尤其14375），不能把不同source的覆盖率直接相减宣称Qwen胜parser；不重新跑NER、猜最近实体、不把basic dependency当完整scope。

在这6个相同输入上，共用确定性的exact citation/重复/typed/literal time-unit库存和unknown validator检查Qwen是否新增真正predicate/argument/condition信息；这些字面检查不是强语义基线。若只做到实体/数字列表，C1失败。此有限仪器实验不声称证明LLM必需；需要同输入强文本方法的直接比较留给下一门。

后续且不在本轮：合格source records seal→原R过去prefix合法事实集合（当前值不可选L）→同records强text baseline→另审semanticattentionrepair两branches→原token风险全封存→最后开评价真值。当前不生成 R、不计算风险/MI/χ、不复跑H019、P006、CoCo或旧E，不自动进入下一门。

## 执行顺序与 stop/go

| 阶段 | 目的 | 固定工作/预算 | GO/STOP |
|---|---|---|---|
| M0 | source identities、2controls、schema/prompt/义务ledger | 0LM；全部6输入hash/模型资产/实际token上限资格，独立source-onlycontrol审查 | 任一控制/信息隔离不合格先停，0generation |
| M1 | 工程实现/资源资格 | 假backend/tiny CPU；实际postimport资源前flight但0LM；installed identity/newclaim | exactsource/预算/计费/资源未齐不得启动 |
| M2 | 固定S-only proposals | 最多6generation requests，≤9216output/actualdecoderforward上界；一次，raw增量保存 | 工程错误封存停；语义/格式失败保留，其余预定请求不自适应改变 |
| M3 | 独立S-only review | 全6source/proposal/义务/controls，0新LM/GT | 全门才允许提下一项prefix-L零LM准入；本轮无自动repair或AUROC |

当前所有阶段均未执行；本proposal只写正式科学合同。代码需独立review给GO后再编写，LM仍由root单独新spec/check/claim/launch/finish。

## Primary边界

source-first extraction和attention steering不是首次：AutoPASTA已从context＋question提关键句再attention steer；UDepLambda/UDepLambda¬已有predicate–argument/negation-scope转换；本轮只检验完整source-record前端是否可用，不宣称方法新颖。[AutoPASTA](https://arxiv.org/html/2409.10790)、[UDepLambda](https://aclanthology.org/D17-1009/)、[UDepLambda¬](https://aclanthology.org/W17-1804/)。引用/JSON正确不证明逻辑scope；当前strong claim是有限instrument资格，不是幻觉检测效果。
