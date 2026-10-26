# myproject_experiment

这里保存 R4 的数据集准备、缓存复验和评价工具。日常检测直接调用
[`myproject`](../myproject/README.md)，不需要经过这些阶段。

| 文件 | 用途 |
|---|---|
| `run.py` | 显式运行下面的五阶段实验命令 |
| `inputs.py` | RAGTruth 官方字段适配、整库文本准备 |
| `datasets.py` | RAGTruth、HalluRAG、ClaimDB 的原始数据适配，分离检测输入与原生标注 |
| `export_hallurag.py` | 将官方 pickle 的文本与原生标签导出为 JSONL，保留来源哈希 |
| `data.py` | 历史 FIT/DEV 划分、文件读写与文件校验 |
| `measure.py` | 读取旧测量缓存或批量运行 observer，导出测量包和进度 |
| `pipeline.py` | 选择 FIT、保存最终逐词分数和身份、封存结果 |
| `evaluate.py` | 分数封存后读取官方标注，计算 AUROC/AP 与 top5% |
| `../tests/` | 数值、模型观测、数据隔离与评价回归 |
| `../tools/verify_frozen.py` | 外部历史参考、原 token 轴和分数逐位复验 |

缓存、测量包、标签、评价报告及文件校验均由这一层管理。它单向调用主线；
主线不依赖此目录。`fit` 只把 FIT 行交给检测器；`score` 只导出最终分数与
原 token 身份，不导出 unary、rank 等内部变量。

在仓库根目录执行，已有冻结测量时可以只用 CPU。以下缓存路径须换成实际外部路径：

```bash
python -m myproject_experiment.run prepare \
  --cache graph/outputs/native_support_ragtruth_all/source_first_v1 \
  --output /tmp/r4-inputs
python -m myproject_experiment.run measure --input /tmp/r4-inputs \
  --task Summary --split train --output /tmp/r4-data
python -m myproject_experiment.run measure --input /tmp/r4-inputs \
  --task Summary --split test --output /tmp/r4-data
python -m myproject_experiment.run fit --data /tmp/r4-data \
  --task Summary --output /tmp/r4-model.npz
python -m myproject_experiment.run score --data /tmp/r4-data --task Summary \
  --partition test --model /tmp/r4-model.npz --output /tmp/r4-scores
python -m myproject_experiment.run evaluate --scores /tmp/r4-scores
```

从官方文本准备时，第一步使用 `--dataset <RAGTruth/dataset> --model <本地模型路径>`。
只有显式传入 `measure --native --device cuda:0` 才会重新运行模型。
输出目录使用新路径，不覆盖已有研究制品。

安装实验依赖与复验：

```bash
python -m pip install -e '.[experiments,dev]'
python -m pytest -q
python tools/verify_frozen.py --packs <历史测量包目录> \
  --baseline <历史R4结果目录> --output <新的验证JSON路径>
```

冻结复验读取外部六个历史测量包和三任务冻结结果，逐位对照包内参考数组、
DEV/TEST 原 token 身份与最终分数；不读取标签、不计算新的 AUROC、不启动 GPU。
完整原始缓存重建仍可使用上面的 prepare/measure/fit/score/evaluate 命令，
缓存和历史结果由调用者显式提供。

## 新数据集测试

HalluRAG 官方文件是 pickle；先按官方 split 导出全部生成器文件，不按模型挑选：

```bash
r4-experiment export-hallurag --files /path/to/HalluRAG/test_*.pickle \
  --output datasets/hallurag/raw/test.jsonl
```

`train_*.pickle`、`val_*.pickle` 分别导出为 `train.jsonl`、`val.jsonl`。
只去掉 `sentence_data` 中的 `internal_states`；原回答、原生句标签和文件/行身份保留。
导出同时写 `<split>.receipt.json`，记录每个原文件和输出的 SHA256，数据适配时将它纳入
manifest。已有输出或 receipt 会拒绝覆盖；若中断发生在 JSONL 发布后、receipt 写入前，
保留原输出，在新的目录重跑。此命令负责转换本地文件，不自动下载官方归档。

主线接口始终是 `HallucinationDetector.predict(prompt, response, source)`。
数据适配器仅整理原始字段，不改变检测算法。标准 `inputs.jsonl` 保存身份、任务、
原提示、实际来源与回答；`annotations.jsonl` 单独保存原生标签。新增文本数据集只需
添加一个返回 `Sample` 与标注的适配函数，并在 `prepare_dataset` 显式选择。

```bash
python -m myproject_experiment.run prepare-dataset \
  --name hallurag --data data/HalluRAG/raw --split test --output /tmp/hallurag-inputs
python -m myproject_experiment.run predict-dataset \
  --input /tmp/hallurag-inputs --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --task QA --device cuda:0 --output /tmp/hallurag-scores
python -m myproject_experiment.run evaluate-dataset \
  --scores /tmp/hallurag-scores --label-granularity token \
  --bootstrap 2000 --seed 17 --output /tmp/hallurag-token-evaluation
```

`predict-dataset` 使用包内冻结参考分布，也可显式传入 `--references`；不在测试集上
拟合。旧 R4 配方经过带标签 TRAIN-DEV 选型，FIT 参考本身不读真假标签；
此处不把两者混称为全流程严格无监督。重复同一命令会跳过已完成回答；输入、模型权重、参考参数或检测代码变动则拒绝
续跑。同一输出目录只允许一个写进程。所有回答评分完成并封存后，才能运行评价。

用预先固定的 ID 名单准备开发或确认子集时，加 `prepare-dataset --ids ids.json`。
文件可以是字符串数组，也可以是含 `ids` 数组的 JSON 对象；例如本次
`data_prep/DEVELOPMENT_IDS.json`、`CONFIRMATION_IDS.json`。输出保留原始文件顺序，
manifest 记录完整请求 ID 名单及原数据/receipt 哈希。重复、缺失、空或非字符串 ID
会报错；`--ids` 与 `--limit` 互斥。不传时保持原全量准备行为。

```bash
python -m myproject_experiment.run prepare-dataset \
  --name hallurag --data data/HalluRAG/raw --split val \
  --ids experiment_results/method_iteration_20261019/data_prep/DEVELOPMENT_IDS.json \
  --output /tmp/hallurag-val-development
```

开发/确认的来源分组由名单生成流程固定；ID 接口只机械选择原样本和原生标注。

| 数据集 | `--data` 所需文件 | 任务与评价单位 |
|---|---|---|
| `ragtruth` | 官方 `source_info.jsonl`、`response.jsonl` | 用 `--task QA/Summary/Data2txt` 选择冻结配方；原生错误字符区间映射至实际 token |
| `hallurag` | 官方导出的 `train.jsonl` / `val.jsonl` / `test.jsonl`；`--split validation` 是 `val` 的别名 | QA；`--label-granularity token` 将句标签扩展至原始 token，直接评价各 token 分数 |
| `claimdb` | `test-public.jsonl` / `test-private.jsonl`、`test_dbs/*.sqlite` | 用 `--split public/private`；原生三分类 claim，当前文本检测器不支持数据库证据 |

HalluRAG 的 source 从生成器实际看到的 `rag_prompt` 提取；不读取仅含答案的 oracle
passage。`undefined` 不作为负例；无法严格对齐的句标注排除并计数，不改写原回答。
主指标 AUROC/AP 及按原始来源组 bootstrap 的区间均保留；HalluRAG 按目标文章聚类，
所有生成器配置一起抽样。句级结果不冒充 token 定位成绩。

HalluRAG 的 token 数据按 [CORTEX §4.2](https://arxiv.org/html/2606.31033v1#S4.SS2)
的核心规则构造：幻觉句内的全部 token 为正例，已知正常句内为负例。
输出 `token_labels.jsonl` 保留每个原始 token 的 ID、文本、字符位置、所属句索引与标签，
不含预测分数；`units.jsonl` 仅将有效标签与原始逐 token 分数配对。没有句级平均或重新分词。
跨句 token 有任何正句重叠则为正；否则碰到未知句或未覆盖字符便排除；不可对齐的回答
全部 token 排除。被排除的 token 仍保存在标签文件中，标签为 `null` 并记录原因。
这几个边界处理是本项目的明确规则，论文未充分说明其实现细节。

这是句标签扩展的 token 伪标签评价，不能检验一句话内部错误词的精确定位。
检测器仍使用冻结 R4；未引入 CORTEX 的监督 MLP。默认 `--label-granularity native`
保留历史原生评价，HalluRAG 此时才使用句内字符重叠加权平均；只作为补充结果。
RAGTruth 已有原生错误区间，token 模式直接使用原生标签；ClaimDB 不提供 token 标签，
显式请求 token 评价会报错。

ClaimDB 可准备与校验，但其证据是完整数据库，需要单独的结构化证据方法。
`predict-dataset` 会在加载模型前拒绝无文本 source 的样本，不生成替代任务成绩。
公开测试有三分类标签，私有测试标签隐藏；评价代码不自动把三分类合并为二分类。
