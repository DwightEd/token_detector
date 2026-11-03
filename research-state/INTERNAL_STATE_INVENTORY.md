# 内部特征保存范围核对

2026-11-03，Asia/Shanghai。核对历史远端目录清单、观察代码、完成记录及已恢复 NPZ 的 header；本次没有联网列举整个云盘，没有读取数组数值或目标标签，没有模型调用。

**结论：远端保存过独立的原向量观测；不能将所有数据概括成“只有分数”。但目前也没有证据表明六个数据集均保存了完整的逐层、逐头内部状态。**

“完整回答人口”“某层全部 hidden 维度”“全部层/头/QKV/注意力”是三种不同的完整性。`hidden[n,d]` 或最终词表 logits 不等于全部层、全部头的计算轨迹。

## 已核实的全量主缓存

| 数据集 | 覆盖与实际核对 | 缓存里有什么 | 完整内部状态结论 |
|---|---|---|---|
| RAGTruth QA | 全900份原观测 NPZ 的 header | `delta[n,2]`、`route[n]`、token/offset/answer/unit等轴 | 该缓存没有 hidden/Q/K/V/attention 字段 |
| RAGTruth Summary | 全900份，同一schema | 同上 | 同上 |
| RAGTruth Data2txt | 全900份，同一schema | 同上 | 同上 |
| FRED FinQA、TATQA | 全356份原观测header；含FIT各32，TEST162/130 | 同样为delta/route及原轴 | 该缓存没有 hidden/Q/K/V/attention 字段；另存目录未知 |
| HalluRAG V3分数 | 已恢复1782份prediction JSON；实际抽核schema | 原token、offset、score及ID/response hash | 分数文件没有内部状态 |
| HalluRAG原生观测 | collection记录1782个NPZ路径/hash，162prefix groups；原NPZ未恢复 | 尚未实际核header；不能从当前采集器代码直接追认旧产物 | 未确认，不能断言远端没有 |

RAG三任务本次2700份header清点记录见 [internal_cache_headers.json](internal_cache_headers.json)。金融356份header由并行只读审计核对；V3金融292份score NPZ也只含逐token阶段分数和轴。

以上是这些已恢复**主缓存**的结论，不是整个远端文件系统的排除性结论。

## 独立保存过内部向量的实验

设远端项目输出根为：

```text
/share/home/tm902089733300000/a903202310/lys/research/token_detector/outputs/
```

| 相对目录 | 数据覆盖 | 代码明确写入的原向量 | 证据级别 |
|---|---|---|---|
| `contextual_probe_20261027/states/` | 104个机制构造例，不是RAGTruth全量 | `source_hidden`、`answer_hidden`；末层post-token最终归一化状态，另含原轴、logp | 历史目录listing＋代码落盘＋实验记录；当前NPZ完整性未核 |
| `delta_local_20261028/observations/` | 196个已暴露RAG回答面板，覆盖QA/Summary/Data2txt，不是2700条完整TEST | `present_hidden`、`absent_hidden`及logp/entropy/rank/token轴；末层post-token状态 | 同上 |
| `source_write_20261028/observations/` | 同196回答面板 | `source_write/history_write/other_write`、`native_attention`、`mlp_write`、`embedding`、`residual`、`rounding`及读出 | 同上；不是每层每头原QKV |

`native_attention` 在上述P006代码里是attention输出向量；source/history写入跨层累加后落盘，不能按字段名字解释成完整attention权重矩阵。

保存代码：`experiments/hidden.py:36–53`、`contextual_probe.py:147–163`；`delta_observe.py:24–32`、`delta_local_run.py:132–151`；`source_write_observe.py:119–124`、`source_write_run.py:108–118`。

## 旧项目中必须继续核实的目录

历史清单标签为 `myPorject/results`（拼写如此），能确认父目录曾列出：

```text
full_qa_message_routes_20261009/
message_routes_20261009/
full_routes_native_slot_witness_20261010/
direct_vector_probe_20261011/
ragtruth_task_extension_20261011/
```

其中确切已见的其他名字包括 `full_routes_canary_20261009`、`full_routes_canary_resume_20261009`、`full_routes_numerical_probe_20261010`，以及 `native_routes_twohop_20261009`。父目录listing没有给出各子目录的数组字段、样本数或层/头覆盖，绝对挂载根也未在本次独立核实。**不能仅凭“full”“route”“vector”这些目录名就断言完整内部状态已经存在；也不能因为本机没下载就断言远端不存在。**

下一步应取这些目录的plan/manifest/collection以及少量原始数组header，确认模型、数据集、样本ID、位置对齐、层数、头数、dtype及seal。只做文件和schema清点，不重跑模型。

## 哪些机制问题已经或尚未回答

- P002隐状态锚距离、P003物理删token依赖、P005post-token方向均有旧实验及负结果，不能换名重复。
- P006确实做了终端residual减source/history写入后重过norm/lm_head的激活消融；不能说项目从未做内部干预。它未证明识别出条件绑定专属电路。
- H016保存312行最终151936维FP16 logits，没有采集逐层hidden或Q/K。其prefix-trie节点是记录单位，不是发现的神经网络核验节点。模型为Qwen3B NF4；历史主观察器为Llama8B，两者不能当同一机制对象。
- 当前尚未完成“当前观察模型在原回答预测位置的绑定专属head/QK定位—路径干预—原token检测收益”证据链。不能把已有内部向量、模型问答正确率或高attention当作该证据链已成立。

关于“回看”的文献边界：ICLR2024 Binding IDs用激活干预支持绑定成分；EMNLP2026的rebinding研究在Gemma/Llama换箱任务中观察到条件信息经过中间位置、在query/readout端再用于绑定检索。这支持研究延迟应用条件，不证明存在通用真假核验节点，也未验证本项目Qwen仪器。[Binding IDs](https://arxiv.org/html/2310.17191v2)，[EMNLP2026正式论文](https://aclanthology.org/2026.emnlp-main.1238/)。后者本次已核实正式venue；旧文献快照“尚未核实录用”仅是当时记录，保持原文件不改写。

后续优先复用可恢复的内部向量；仅当需要的内部量确实未保存时，才提出补采同一冻结观察模型的必要forward。读取模型内部量需要运行被研究模型，不需要另请一个问答模型解释这些向量。所有主线结论仍必须落到已有回答的幻觉检测，而非新增角色问答榜单。

证据来自本地恢复根 `D:/projects/research/outputs/cache-recovery-20261102/` 的历史listing、`fusion-cache-20261102/` 的实际缓存及仓库观察代码。私人下载日志和认证材料不发布。
