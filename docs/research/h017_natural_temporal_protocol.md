# H017 M4：自然时间范围错例的原轴观测准入（编码前）

本阶段只补采一条固定自然回答的匹配观测，不产生检测分数，不检验因果必要性。父任务在来源事实表先冻结后，单独查看既有回答，识别 15716/source15540 的 `over 300 years BC` 被写成 `around 300 years ago`。错误是时间参照/范围表达，不是 Castor/Pollux 的主体互换。该例及所属 panel 已暴露，不能作为新确认集。

## 身份、许可与研究问题

仅用既有 RAGTruth QA TEST 15716/source15540 的完整原 Q/S/R；不编辑来源、问题、回答，不重分词修复，不补充问句、替代单位、答案或外部神话知识。来源许可 JSON SHA256 `23e77b9ac9cb2703b14e49cd61b62a910f008ab865e3adce18e7e8e03f636f0a`，独立 review SHA256 `ec8a97b5dce7e9c2c2f1c1a6caf735240083a633f99c81cd373f6dcfe0763229`，source SHA256 `485cc2e00a43df28446cbed4a383f6c63bd8aa7066f6e3fc538acd258fedc186`。原 Q 是 prepared observer input，不承诺恢复了原生成器当时的完整提示。

新问题：在这个真实时间参照误表达中，冻结 Llama3.1 观察器对原 token `300 / years / ago` 的绝对条件概率，以及 source 数字、后置 era marker、完整事实闭合位置的全部层/头访问轨迹分别是什么？值片段可用性和完整条件可用性不能混作同一位置的表示。

因果 decoder 中 source300 的 K/V 不能依赖更晚的 sourceBC；这是架构性零可用性，不是新实证机制或创新证明。BC/事实闭合及以后的位置可以接收该条件，是否真正编码/利用它仍未知。尤其 R query 可从较早层、其他 source keys、prompt 或先前 R 已整合该条件；当前层只关注300不等于没有条件语义。所有结果均限于观察器 teacher-forcing 原回答，不是 Mistral 原始生成原因。

## 与关闭实验的实质区别

| 已关闭方向 | 本次不同的新信息及不重做的部分 |
|---|---|
| P005 post-token direction | 不计算 source/absent 位移、whitening、cosine 或风险；补充同一完整输入的全部32层、全部77原query、原token绝对概率及完整 postRoPE KV。 |
| P021 relational null worlds | 没有 source shuffle、absent/null 世界或混合；只有一个原样 native 世界。 |
| P026 repair / prefix reset | 没有改写 BC/ago、year repair、未来交互、重置或移除回答历史。只观察既有全程，不将 post-token 稳定称作纠错信号。 |
| H017 M3 | M3 正常简例未实例化高概率错误，主 K-role 门已否定；本次换到来源先许可的真实时间错例，新增 full-layer 匹配记录，不用正常 toy 的新头/层/prompt 救援。 |

这些区别只构成一次受限观测的准入，不构成新方法、检测有效性或文献新颖性。

## 一次原模型和一次完整 head 的冻结预算

- 仅 1 次 native 完整32层模型 forward，batch1，完整441原tokens；不 early-stop，不 source-block，不 cached-query 分支。
- 原轴为 P365、T76、N441。`query_positions = range(P-1,N)` 共77行。全部77行 Q/native cos/sin/native o_proj INPUT headAV 在全部32层捕获；全部32层 postRoPE K/V 对完整441位置保留。
- final norm 后 hidden 的 `query_positions[:-1]` 共76行一次送入未分块128256词表 head；精确预测原76个 answer_ids。第77行是最后R token之后，没有观察到的下一token，不为其虚构概率。
- 保存全部76×128256 FP32 logits、对应 hidden、原 token ID/原R字符 offsets、每个原token的 logp/p/rank/top1/entropy。p>0.5 仅为“该原token绝对概率超过一半”的描述性标记，不是风险或正确性阈值；不以二候选重归一概率代替。
- 全部32层×32物理Q heads×77 receivers×441 keys 的 attention 保留，不挑 head/layer/branch。所有原R位置都保留；`300/years/ago` 只按固定原字符串和原offset锚定报告，不读取发布错误span。
- 一次失败即留存并停止，无自动重试、换precision、缩输入、改单位或增加forward。计数在调用开始前递增，异常也计入attempt。

## 原轴与 source-only 锚点

复用 M2 `reconstruct_alignment` 的独立 source/response mainline encoders 和 exact integer axes；不得 joint-retokenize。必须与当前 M2已合格15716结果、M1原capture witness、prepared inputs/manifest、record及arrays SHA 完全绑定。原token_ids、answer_ids、source_mask、query_positions四轴逐值相等；P/T/N不符即在模型前停止。

source 字符区间均在原 decoded source 中，半开 Unicode code-point：

| 锚点 | 区间与原文字面 |
|---|---|
| G03完整事实 | [138,244)：许可历史引用句，含结尾句号 |
| G03时间范围 | [226,243)：`over 300 years BC` |
| 数字300 | [231,234)：`300` |
| era marker | [241,243)：`BC` |
| 完整G03闭合字面 | [243,244)：`.`；只是可用完整句前缀的位置，不假定语义闭合算法 |
| 无关G06量/范围 | [550,593)：`the third 30 degrees of the zodiacal circle`；其中[560,562)为`30`，不视作同量纲合法替代 |

每个锚点输出 exact substring、所有相交 token 的原绝对位置/ID/offset，以及 contained/boundary_overlap/unknown。不得把跨边界token强行切成精确语义token；所有 overlap 及其影响保留。BC和300须能区分严格先后，未映射/次序不明时该局部对比unknown，不修补。还要保留 era_first_token 和 era_complete_token，以覆盖marker多token情形。

固定描述性profile同时保留300、BC、G03、G03closure、G06角度、全部source、全部source且position>=era_first、全部source且position>=era_complete、其他prompt、earlierR、self。跨锚点profile允许重叠，不能相加充当partition。完整 raw key axis 支持审计，不把“没看BC token”误写成“没有任何可携带BC的输入”；era以后全部source key、prompt/R路径都可能携带该信息。

## 资格与函数复用

新 runner 仅 `experiments/natural_relation_observation.py`，tests 仅对应新文件；父任务接入 central CLI/spec/claim。新 runner 不复制模型数学或另设attention kernel。

1. 使用未改的 `verify_assets`、`capture_identity`、`head_prerequisite` 和 `StreamedObserver`。原模型9资产、BF16存储/FP32 Linear与embedding输出/native算术、Transformers4.57.1 SDPA、无AMP/TF32、原rotaryFP32等照旧；资格函数的源码/包整体身份绑定。
2. 使用 `frozen_capture_functions(capture_root)` 仅 AST 提取既有hash-pinned producer；`capture_selected_layer(native,15,P-1,stop_after_layer=False)` 原样捕获。不得导入整个旧package或修改producer/hook。用现有 native dtype guards 在历史 `.float()` 之前验证FP32。
3. 在这同一次 native full forward 内，将 block15 native nodes三个site、preRoPE Q全部77行、preRoPE K/V全部441位置与旧native世界FP16 archive逐坐标比较。**直接调用未改 `coordinate_summary`**，沿用FP16 interval +0.002+0.0001abs(saved)固定policy；32Q/8KV逐头保留，零违反才qualified。旧blocked世界既不重采也不声称本次验证。
4. `ExitStack` 对全部32层复用现有 `capture_rows(observer,rows,layer=...)`，取得实际native Q/cos/sin/o_proj-input；调用现有 `attention_diagnostic` 对每层全77行以native RoPE、contiguousGQA、absolute query因果mask和真实postRoPE KV重放。全层全部headAV坐标固定误差界0.002+0.0001abs(native)，任何一层失败则整个观测未合格。原old raw M1不能替代新phase检查。
5. 全部raw states、原轴、原layer15 archive比较数据和final hidden在任何数值资格gate/head前落盘；completed forward不能因后续gate失败被丢弃。每层attention重放结果增量保留。head调用只在完整native身份及所有AV gates通过后进行；否则head计数0。后续失败也保留已算logits/endpoint并记录failure，无重试。
6. 结束核对hooks清理、weights全meta、asset/input/dependency hashes未变和实际计数；每个完成阶段封存，最终seal不覆盖历史partial evidence。

## 资源准入

复用有界streamer，唯一变化为显式 `ResourceLimits(max_sequence_tokens=512)`。GPU计划上限6GiB、host single mapped tensor上限1GiB、GPU/RAM剩余floor各512MiB、workspace256MiB均不变；不关闭用户进程。序列实际441，无长上下文扩展。

441位置全部32层KV约110.25MiB；32层77行Q+headAV约77MiB，cos/sin约2.4MiB；全attention约132.7MiB；76行完整logits约37.2MiB，加原layer15坐标与small hidden仍可按层落盘并释放临时对象，避免把重复原始tensor/诊断同时常驻。KV应一次归档后逐层读取CPUview，attention诊断每层后立即保存/释放副本。计划原模型约一次完整权重流16.06GB及一次head权重重载；记录实际walltime、峰值GPU allocated/reserved、进程RAM/private/peakcommit、transfer bytes，而不把缓存后处理时间当全部成本。

只做 tiny official CPU synthetic tests：77/76语义对应、strict原轴与spans/overlap、full-call/head计数、无未来R泄漏、pre/postRoPE与GQA/因果mask负控制可复用既有 tests；额外覆盖多层capture/cleanup、旧FP16全部坐标gate、失败raw保留/不重试。此阶段不由子agent运行真实tokenizer/原权重probe；父任务审核后claim唯一实际观察。

## 竞争预测与停止口径

| 可能观察 | 允许的解释与禁止升级 |
|---|---|
| 错误的300/years/ago之一或全部p很低 | 该观察器并未高概率续写这些原tokens；停止宣称“高置信错绑”，不换prompt/model救援。 |
| p高、数字attention明显，而BC/closure profile低 | 与值片段访问相容，但无法证明条件未表示/未利用；query/其他keys的前层路径仍可能携带BC。 |
| p高，同时BC/闭合/era后位置也有访问 | 条件可用位置被访问与错误输出并存；不能把来源聚焦等同正确性，也不能据attention宣称模型理解后故意丢条件。 |
| 原词写入后profile变化 | 只描述teacher-forced轨迹；没有反事实，不称自我强化/因果持续/回看验证节点。 |

这些不是互斥可识别机制类，也没有“挑一个最像机制”的决策树。本次产物是全轴可审计观测与明确unknown，不是模型语义读出或risk。任何失败全量报告，不按好看的层/头/词缩小报告分母。

直接文本的BC/ago时间参照核验已可指出该例问题；本次不能证明白盒必要性、定位更可靠、更便宜或六任务AUROC改善。原生成器状态、source-reading Q/hidden、受控必要性、自然覆盖率均仍未知。后续如要进入算法，需要另行冻结可自动获取的语义范围接口、原回答risk投影、强文本核验对照与自然数据评价；本阶段不自动授权扩大模型预算。
