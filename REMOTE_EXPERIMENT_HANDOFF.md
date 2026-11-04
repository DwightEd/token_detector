# 换电脑继续实验：远端数据位置与执行交接

更新：2026-11-03，Asia/Shanghai。代码仓库：<https://github.com/DwightEd/token_detector>。

**2026-11-04 本机恢复完成：** 新下载的指定材料位于
`D:/ResearchData/token-detector-remote-20261104/`，共 **34,114 文件、27,086,397,672 字节**。
包括三任务 TEST2700 原生包及450份来源、QA TEST900 第15层观测及冻结代码/配置、
Summary/Data2txt TEST各900的扩展观测、P002/P005/P006完整小面板及两份RAGTruth原文件。
所选文件的原始哈希、大小、封存引用与数组头已核验；QA900共11,700个数组头检查通过。
下载不解码标签或数组值、不运行模型，也不改写历史 seal。
原有六任务输入、主观测和历史分数另保留在
`D:/projects/python_projects/research/token_detector_data/`（20,399文件、159,669,343字节）。
本次收据为 `outputs/cloud-download-20261104/completion.json`；数据根的
`00_manifests/local-acquisition-20261104.json` 保存同一索引。下文 C盘30.82GB为另一台电脑的
历史恢复范围，不能代替本机清单。未下载模型权重或197GB未完成归档；文档中的H017近期原始输出
在对应云端目录在线列举为零，不能视为已恢复，公开结果仍可从Git阅读。

**续接目标已纠正：** 请先读 [检测目标与实验验收](research-state/DETECTION_OBJECTIVE.md)。H016角色问答属于已关闭的辅助诊断，不能当作新方法的幻觉检测成绩，也不要在另一台电脑继续扩展这条问答支线。

**本轮按用户指令优先 RAGTruth。** 本轮所选文件已全部下载并核验，含15个前缀及legacy联合目录复验、QA900的11,700个数组头检查。最终锁定范围为 30,817,905,041 字节（约30.82 GB）、51,812 文件，统一位于 `C:/ResearchData/token-detector-remote-20261103/`。`00_manifests` 保存分类说明、不可变清单、实际哈希与核验收据；不把传输中或未核文件计为已恢复。最终收据见 `outputs/remote-complete-download-20261103/acquisition-finisher-v2-final.json`；收尾器已退出，不需重启下载。数值RoPE重放和新检测实验仍未执行。

**内部状态另存清单：** 见 [内部特征保存范围](research-state/INTERNAL_STATE_INVENTORY.md)。已找回并核验 P005/P006 小面板向量；另外在线确认 QA TEST900 的 block-15 坐标缓存。Summary/Data2txt 已确认的是原生标量和局部注意力扩展，尚未找到同范围原 QKV。旧197GB full-QA目录实际只有 DEV1008/FIT1295、TEST0，是未完成运行；本轮仅保留其根运行元数据，不能将其当成 TEST900 内部状态。勿因主缓存只有标量就重跑模型，也勿因目录名带 full 就宣称已有完整逐层逐头状态。

**先恢复已有观测，不启动新模型实验。** 下方六任务清单保留作完整研究交接，本轮不扩大下载到其余数据集。H016 已完成，不需要另一台电脑重跑。六任务各 AUROC ≥ 0.8 的目标尚未完成，当前没有可直接宣称达到该目标的新方法。

### 本轮 RAGTruth 内部观测的准确位置

以下路径均相对于 `$BASE/research/`；存储对象路径去掉 `/share/home/` 挂载前缀。

| 远端目录 | 本轮选择与字段 |
|---|---|
| `graph/outputs/native_support_ragtruth_all/source_first_v1/` | 根据其 manifest 与当前三任务 TEST ID 对齐，只选各900个完整 `responses/<index>/` 文件包、450份来源及必要参考；原生标量不等于 QKV |
| `graph/outputs/supervised_local_transport_20261008/` | 当前 QA TEST900 的 `capture/<answerID>/arrays.npz`、`record.json`，及协议/冻结源码；block15回答T+1行的residual/write、pre-RoPE Q和全输入pre-RoPE K/V。阻断world的hook作用于执行的0–15层，非source query禁读source key，非15层单独消融；early-stop无最终logp。采集不拟合标签，目录名不代表原观测受监督训练 |
| `myPorject/results/ragtruth_task_extension_20261011/` | Summary/Data2txt 当前TEST各900份 measurement NPZ/JSON及参考/协议；`attention[T,32,8]` 为第15层局部回答历史注意力，不等于直接来源attention，不含完整hidden/QKV |
| `token_detector/outputs/delta_local_20261028/`、`source_write_20261028/` | 已暴露196回答小面板的末层hidden/分区写入向量；不是三任务全量TEST |

精确采集清单与实际文件校验分开：在线文件存在、采集配置的计划人口、当前已下载文件、原封存引用吻合、头信息字段覆盖分别记录。模型权重暂不下载；真实标签只在方法和评分全部冻结后用于最终评价。

## 1. 存储入口和路径的含义

存储页面：<https://console.aicloud.szu.edu.cn/console/filestorage/detail/tm902089733300000?role=0>。

历史运行记录确认的 Linux 路径：

```text
BASE=/share/home/tm902089733300000/a903202310/lys
PROJECT=/share/home/tm902089733300000/a903202310/lys/research/token_detector
```

下表中的 `$BASE`、`$PROJECT` 都指上述实际路径。早期对话叫过 `myProject`，但恢复的运行记录使用 `research/token_detector`。

存储下载记录还使用这种相对路径：

```text
tm902089733300000/a903202310/lys/research/token_detector/...
```

这是存储对象的路径表示；`/share/home/` 是实例中的挂载前缀。本轮已在线核实 RAGTruth 所选路径并逐文件传输；控制台目录不能展开时，可使用文件存储的下载接口，依据精确 manifest 和现有授权会话取得文件。不要凭目录名扩大到整个旧实验归档。下方其他数据集的历史路径保留，未纳入本轮下载。

## 2. 优先下载：继续六任务实验的最小材料

| 材料 | 远端位置 | 用途与范围 |
|---|---|---|
| RAGTruth QA 输入 | `$PROJECT/outputs/ragtruth_full_shared_20261029/test_QA/` | 全部 900 答 |
| RAGTruth Summary 输入 | `$PROJECT/outputs/ragtruth_full_shared_20261029/test_Summary/` | 全部 900 答 |
| RAGTruth Data2txt 输入 | `$PROJECT/outputs/ragtruth_full_shared_20261029/test_Data2txt/` | 全部 900 答 |
| 金融输入 | `$PROJECT/outputs/finance_transfer_20261028/fit_finqa/`、`fit_tatqa/`、`test_finqa/`、`test_tatqa/` | 原 FIT 与 TEST；TEST FinQA 162 答、TATQA 130 答 |
| HalluRAG 输入 | `$PROJECT/artifacts/final_package/hallurag/prepared/` | 全 1782 答；用发布的 answerable/target-available 元数据单列主组 891 答，保留另外 891 答 |
| RAGTruth 原生观测 | `$PROJECT/outputs/ragtruth_full_shared_20261029/native_QA/`、`native_Summary/`、`native_Data2txt/` | 原 token、unit 与观测缓存；优先复用 |
| 金融原生观测 | `$PROJECT/outputs/finance_transfer_20261028/native_v2/` | 64 FIT + 292 TEST 的原观测 |
| HalluRAG V3 原生观测 | `$PROJECT/outputs/matched_local_20261031/hallurag_native/` | matched-SDPA 观测 |
| 五组 V3 历史分数 | `$PROJECT/outputs/persistent_witness_v3_20261029/` | 整目录；包含 `scores_QA/`、`scores_Summary/`、`scores_Data2txt/`、`scores_finance/` |
| HalluRAG V3 历史分数 | `$PROJECT/outputs/fixed_persistence_20261031/hallurag_scores_all7/` | 下载外层封存记录及 `witness/` 全部载荷；其中 `full_gap/` 不等于 R4 |

prepared 目录至少保留 `inputs.jsonl`、`manifest.json`、`annotations.jsonl` 及原有 seal；native/score 目录保留全部 `plan.json`、`collection.json`、`complete.json`、原 NPZ/预测载荷和 `seal.json`（以各目录实际清单为准）。不要只下载分数 CSV 或汇总表，也不要打散目录。

**推断只读取 inputs 和所需的非标签配置；annotations 保留到方法、参数、人口和全部分数冻结后的评价阶段。** 不按已知错例筛选数据，不更换原 token 轴，不把未覆盖项当成低风险。

## 3. 原始数据与模型

| 材料 | 远端位置 | 说明 |
|---|---|---|
| RAGTruth raw | `$BASE/data/RAGTruth/dataset/source_info.jsonl`、`response.jsonl` | 恢复原问题、source、response 与发布标注；优先使用现有 prepared 保持人口一致 |
| FRED raw | `$PROJECT/outputs/finance_transfer_20261028/raw/test_finqa.csv`、`test_tatqa.csv`、`ACQUISITION_RECEIPT.json` | 发布的合成编辑数据；不能补造缓存里缺失的原生成问题 |
| HalluRAG raw | `$PROJECT/artifacts/final_package/hallurag/raw/test.jsonl`、`test.receipt.json` | 同时保留 answerable 元数据 |
| 历史观察模型 | `$BASE/models/Meta-Llama-3.1-8B-Instruct/` | 需要新 GPU 观测时下载整个模型目录，包括全部 shard、index、config、tokenizer |

两份 RAGTruth 原文件的历史 SHA256：

若需从 raw 重建金融 FIT，再备份同一 `raw/` 下的 `train_finqa_3k.csv`、`train_tatqa_5k.csv`；已有 prepared FIT 可直接使用。

```text
source_info.jsonl  0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b
response.jsonl     e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073
```

原 RAGTruth 观察记录缺失完整权重与 GPU backend 身份，不能承诺换机后精确重现历史数值。finance/Hallu 的部分权重身份已核对，具体边界见 [数据合同](research-state/DATASET_CONTRACTS.md)。8B 历史观察和本轮 3B NF4 观察属于不同仪器，不直接替换后当成同一对照。

## 4. 近期本地实验：需另行转移，不能假定已在云端

以下文件在本机生成，**远端位置未确认**：

```text
D:/projects/research/token_detector/outputs/h006-20261103/
D:/projects/research/token_detector/outputs/h007-burden-20261103/
D:/projects/research/token_detector/outputs/h015-prefix-trie-20261103/
D:/projects/research/token_detector/outputs/h016-paired-scope-20261103/
```

H016 的完整封存观察、评价、32 个原始 NPZ 和报告已经复制到：

```text
D:/Desktop/myproject_研究资料_20261102/15_H016_Paired_Scope_20261103/
```

把这个桌面目录整体复制到另一台电脑即可查看原始材料；入口为 `START.html`，正文也有 Markdown。封存观察与评价合计约 105 MB。Git 只发布代码、报告和小型证据，不包含这些完整缓存或模型。

H006 的本地模型为 `D:/projects/research/models/DeBERTa-v3-base-mnli-fever-anli`；其远端目录未确认。冻结身份见 [H006 model identity](research-state/protocols/H006_nli_model_identity_v1.json)。H015/H016 使用本地 `D:/projects/research/models/Qwen2.5-3B-Instruct`，也不能推断云端同名目录存在。

**已完成而不要重复的工作：** H015 两输入工程资格共 16 前向；H016 compound/atomic × independent/joint 共 312 前向、86,930 输入 token，完整四臂分数先冻结再评价。角色正确数分别为 9/16、16/16、15/16、16/16；仅是 8 来源的已暴露构造诊断，不是六个自然任务达到目标。详见 [H016 报告](docs/results/h016_paired_scope.md) 和 [恢复状态](research-state/RESUME.md)。旧 H013 严格概率门失败保留，不能重封成成功。

HalluRAG 历史 R4 逐 token 数组已知缺失，目前恢复的是汇总报告，不能承诺从上表下载。不要拿 `full_gap` 冒充它。

## 5. 另一台电脑的执行顺序

1. 拉取本仓库，先阅读 [AGENTS.md](AGENTS.md)、[RESEARCH_BRIEF.md](RESEARCH_BRIEF.md)、[RESUME.md](research-state/RESUME.md)，运行 `python tools/research_memory.py context`。本轮 H016 已结束，勿根据旧状态再次启动。
2. 下载第 2 节的完整目录并保留相对结构；如有新的 GPU 观察需求，再准备第 3 节模型。记录本机根目录到历史路径的映射；**不要修改 sealed plan 中的历史绝对路径或改写原 seal 来迁移**。需要迁移的执行入口应显式接收新路径。
3. 新建独立 Python 3.11+ 环境，安装项目及实验依赖。依据新机器实际硬件验证 CUDA/torch；旧机器的 GPU、量化和数值身份不能直接照抄为新机器的事实。
4. 优先验证/复用已有缓存；新观察先冻结协议、全部样本、资源预算、源代码与模型身份，登记 run，核对已有请求，避免重复推理。
5. 所有评分封存后再评价。返回完整输出目录、命令、环境/模型身份、错误日志和原 seal；不要只返回一个 AUROC。

安装及命令入口检查：

```bash
git clone https://github.com/DwightEd/token_detector.git
cd token_detector
python -m venv .venv-research
# Linux: source .venv-research/bin/activate
# Windows PowerShell: .venv-research/Scripts/Activate.ps1
python -m pip install '.[experiments,probe,dev]'
r4-experiment --help
r4-experiment transfer --help
python tools/research_memory.py context
```

下面是**现有 R4 的新运行示例**，不是 H016 自然数据推广，也不承诺历史数值复现。`STORE_PROJECT` 和 `LLAMA_MODEL` 改为新机器的下载位置；使用一个全新的输出目录：

```bash
STORE_PROJECT=/path/to/downloaded/token_detector
LLAMA_MODEL=/path/to/Meta-Llama-3.1-8B-Instruct
r4-experiment predict-dataset \
  --input "$STORE_PROJECT/outputs/ragtruth_full_shared_20261029/test_QA" \
  --model "$LLAMA_MODEL" --task QA --method r4 \
  --output outputs/otherpc_qa_r4_v1

# 评分完成、封存后才执行：
r4-experiment evaluate-dataset --scores outputs/otherpc_qa_r4_v1 \
  --label-granularity token --output outputs/otherpc_qa_r4_eval_v1
```

已有观测的集成入口是 `r4-experiment transfer score` / `transfer evaluate`；V3 显式方法名为 `witness`。先看各子命令 `--help` 并使用相应冻结协议，不能把以上 QA 命令的 task 名简单改成 FinQA/HalluRAG 来假装完成六任务实验。现阶段还没有通过自然数据资格的 H016 全任务方法或一条“一键六任务 0.8”命令。

## 6. 路径依据与已知缺口

本清单经读取历史 metadata 核对，没有重新登录控制台验证在线存活，也没有读取目标 gold 内容：

- `cache-recovery-20261102/remote/native-metadata/native_{QA,Summary,Data2txt}/plan.json`：prepared 路径与 Llama 模型路径；`native_v2/plan.json`：金融四组 prepared。
- `fusion-cache-20261102/witness/scores_{QA,Summary,Data2txt,finance}/plan.json`：各原生观察目录；`hallu_native_metadata/plan.json`、`hallu_v3_scores/plan.json`：Hallu 输入与观察。
- `cache-recovery-20261102/bita-download-persistent-status.json` 及下载记录中的纯路径：V3 历史分数存储位置。私人下载日志不发布。
- 本地 `outputs/full-input-inventory-20261103/inventory.json`：原始数据路径、SHA256 与 3,883 个完整输入人口；[DATASET_CONTRACTS.md](research-state/DATASET_CONTRACTS.md) 保存可公开的数据/观测差异。

上述本地恢复目录的根为 `D:/projects/research/outputs/`，不随 Git 发布。下一台电脑只需按上表下载数据；查不到某项时记录缺项，不猜路径、不改人口、不重建历史标签。
