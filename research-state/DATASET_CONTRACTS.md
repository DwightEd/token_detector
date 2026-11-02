# 六任务数据与观测合同审计

本次新增的是输入、来源/生成器队列、原始 scorer unit 和观测文件身份的核对。没有重算已有 FN/FP、AUROC 或 detector scores，没有调权、调参、新建 LM 观测。完整统计与证据 SHA256 见 [dataset_contracts.json](dataset_contracts.json)。

## 可直接核实的新差异

| 任务 | 回答 / source | 来源字符中位数 | 回答有效 token 中位数 | 含数字 token | scorer unit token 中位数 / P90 |
|---|---:|---:|---:|---:|---|
| RAG QA | 900 / 150 | 1254.5 | 126.5 | 2.85% | 20 / 36 |
| RAG Summary | 900 / 150 | 2520.5 | 124 | 1.89% | 23 / 36 |
| RAG Data2txt | 900 / 150 | 2299.5 | 180 | 2.18% | 21 / 32 |
| FRED FinQA | 162 / 142 | 4053.5 | 85.5 | 19.69% | 22 / 54 |
| FRED TATQA | 130 / 110 | 1390 | 40 | 19.85% | 24 / 51 |
| Hallu answerable | 891 / 81 | 1419 | 30 | 4.89% | 未恢复 scorer unit_index |

“含数字”仅指原始 token 字符中出现 ASCII 数字，包含年份、地址、编号等，不等于数学事实或错误标签。scorer unit 使用缓存 `unit_index`；Hallu 的原生标注句长另记为中位23、P90=35，不能代替其未恢复的 scorer unit 长度。

1. **文件集合 ID 的差异不是不同权重的证据。** RAG / finance / Hallu 记录7 / 11 / 13个 model files。RAG 的7个 JSON metadata 与其余两组逐 hash 一致；finance 与 Hallu 共有的11个文件全部一致，含4个 weight shards。Hallu 多 `params.json`、`tokenizer.model`。RAG 没有原始 weight shard hashes，其计划明确限制为“历史 primitives；原始 GPU source/weight hashes unavailable”。因此可以确认共同记录一致，不能补出 RAG 未记录的权重身份。
2. **finance 与 Hallu 确实换了观测执行路径。** finance 记录的 `model.py/source.py/observer.py` hashes 与保留的 Git `be1b8a3d53703490f5ed01ec899dbf895fce77b1` 精确匹配：BF16；source-present full 使用 eager，其余 prefill/local/source-absent 使用 SDPA。Hallu 的原生 plan 明确为 `matched_sdpa`，且 native seal 被 score plan 绑定。RAG 的 plan.source 则属于 `collect_cached` 重封装阶段，不能代表旧 GPU 观测代码或 backend。
3. **Data2txt 的实际序列化与指令名称不同。** 900/900 来源是可解析的 Python dict literal，0/900 是 strict JSON；指令统一称其为 JSON。所有记录有相同9个顶层字段，包括 `hours/attributes/business_stars/review_info`。FRED 两组来源全部含 prose 中的嵌套数组文本。这些是 observer 实际收到的不同格式，格式影响尚未由同事实对照识别。
4. **finance 的回答更短，unit 尾部更长。** FinQA/TATQA 的 scorer unit P90为54/51；RAG三任务为36/36/32。有效长度恰为64的 units 分别为46/610、22/277，RAG为15/5797、4/5124、10/8247；这些计数不直接等于强制截断次数。含数字 token 比例约19.7–19.9%，RAG为1.9–2.9%，Hallu为4.9%。回答总长、unit 内上下文长、数字结构是三个分开的变量。
5. **Hallu 的格式与队列有明确结构。** 主要组81篇 source × 11个发布生成器/量化配置；Llama2-13b仅有3个量化标签配置，Llama2-7b和Mistral各4个。3个 prompt templates × 1/3/5 chunks 的9格各99答，计数上交叉平衡；每篇 source只出现在一格，没有同文章跨格式配对。567个Llama2回答全部保留前导空白，324个Mistral回答全部没有；其他五组均无前导空白。这是字节/生成器队列的关联，尚未确定来自原始生成还是发布后处理。

## 既有合同的本次核对，不冒充新现象

- RAG QA保留150个实际问题，每题6个生成器；其source均含3个passages。Summary保留33–200词的原始长度指令，中位上限102词；Data2txt统一要求100–200词。无问句在这两项生成任务中不构成缺失QA信息。
- FRED FinQA/TATQA的162/130个输入都使用固定 `Reference material: <SOURCE> Response:`，原始生成问题和generator在缓存中不可用；`not_released`不可解释成同一个已知生成器。原生标签明确是synthetic/silver edit spans，而不是自然模型错误的人类逐事实标注。
- FRED按精确reference文本SHA256聚类：142/110个test references；其中18/16个reference有2个回答，1/2个有3个回答。FIT与TEST精确reference重叠为0，company/filing身份没有记录，不能推出公司级独立。
- Hallu保留实际问题内容；发布消息contents被连接为文本，再由observer的chat template作为单个user消息编码。内容保留与原始角色结构保留是两个条件。主要组每article一个question/prefix，pooled组每article两个不同retrieved contexts，共162个prefix groups。
- RAG提供错误字符spans；FRED提供edit字符spans；Hallu提供原生句判断再广播到相交token。字符span、原生句、scorer unit三种单位分别报告。旧的FP位置统计与句级广播现象只引用既有封存诊断，本次未重算。

## 确定性分层

**verified：** 上面的文本、轴、队列和逐文件hash差异均由缓存直接核对；2992个RAG/FRED native records的payload hashes通过，Hallu的1782个prediction payloads与seal核对后取answerable891答。新下载只有3个Hallu native metadata文件，没有模型或观测NPZ。

**associated：** 数字密度、问题缺失、合成编辑、unit长度尾部、prompt格式、生成器队列和backend共同随数据集变化。它们是具体的竞争解释。Hallu前导空白与生成器完全关联，旧FP位置集中与它共存，尚不能据此断言去空白会修复误报；原始回答字节和token身份继续保留。

**unidentified：** 这些差异各自造成多少AUROC变化、原始RAG GPU权重/backend、FRED原始问题/generator/issuer，以及Hallu实际scorer unit分布仍未识别。Hallu历史plan也没有`context_budget`字段，不能把较新代码中的限制追认成历史运行保证。

## 新的辨别试验：同一回答的前导空白干预

这是尚未执行的观测接口 **qualification**，不作为新方法。567/324的队列关联无法区分生成器与前导空白；新的识别信息应来自**同一回答**的受控改变。

- 固定source、question/prompt、observer/checkpoint/tokenizer、backend/BF16/hardware及现有读出规则。对全部1782个原始回答保留original、统一前置一个ASCII空格，以及有前导ASCII空格时删除一个、没有时保留原样的配对版本；891个answerable主要组继续单列。事实词和标点不变，不选择既有FP案例子集。
- 按精确偏移把原始内容字符映射到每个版本。BPE token数、ID或unit边界可能改变，必须记录这些映射并比较相同的原始首个内容词/片段及完整内容曲线。原生句/字符标签仅作坐标平移，unknown原样保留；不重标、不删首token、不丢改变分词的回答。
- 设置**长度/格式匹配的非起始空白对照**：在预先声明的内部空白边界增删相同数量的ASCII空格，保持事实字符序列。对照位置仅由冻结的tokenizer规则选择，尽量匹配字符数与总token/unit格式变化，不看risk或gold；无法精确匹配的回答仍保留，单列unmatched-control层。
- 观测端点是同一原始内容位置的配对present/absent logp、contrast与现有final变化，再比较前导干预和内部匹配对照。分词/unit变化作为可能的中介报告。已有位置计数不再作为这个新试验的结果；标签分层诊断只在所有版本、人口和分数冻结后进行，旧人口仍属探索性。

只有超出匹配对照的前导位置特异变化，才支持generation→observer接口参与首内容位置告警的解释；没有这种变化则不支持这一具体归因。该资格核验本身不说明应去除空白，也不构成新评分法或六任务0.8保证。Finance数字密度与unit尾部仍保持associated结论，等待其各自匹配控制。

## 其他归因所需的受控干预

1. **先建立观测桥接。** 固定checkpoint shards、tokenizer、BF16/hardware、原始source/prompt/answer字节、response token/unit轴和readout，在同一小组预先声明的无标签输入上，只切换legacy present-full eager与两侧matched SDPA。比较配对原始logp/delta及固定读出排序；所有新分数冻结后再读取最终真值。这样识别backend作用，不能代替数据生成机制归因。
2. **分别核验格式和条件信息。** 同一Data2txt record只改变Python literal/strict JSON序列化；同一Hallu article/question/answer只改变原始roles/flattened wrapper。每次固定其他条件，不做混合权重或参数搜索。FRED原始question与generation recipe需要先恢复，不能用猜出的question构造对照。
3. **生成机制与标注目标需要匹配比较。** 若要把落差归因于synthetic edits或natural generation，需要同source/task/generator的明确生产记录和共同的事实命题评价合同；新增标注仅承担方法与分数冻结后的最终测量。当前六组不同source与不同标注粒度不能单独给出这种因果结论，也不能保证某个模块让六项都达到0.8。

以上控制均未执行。已知错误计数继续复用；下一步实验应回答这些尚未识别的问题，并把已反复接触的人口保留为探索性证据。
