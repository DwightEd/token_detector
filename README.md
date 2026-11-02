# Token detector

逐 token 的 RAG 来源风险检测与研究工具。输入为完整提示、其中实际出现的来源文本和已有回答，返回原 token 的位置与分数。

研究主线固定为**无监督检测，真值仅用于冻结后的最终评价**，不用于训练、调参、选方法或风险校准。完整约束见[用户确认记录](refine-logs/theory-grounding-20261102/USER_CONSTRAINTS.md)。

本版本接回 2026-10-31 已核验的远端 v7 源码：`myproject/` 为主库，`experiments/` 为统一实验入口，`experiments/research/` 保存研究候选。[恢复与验证记录](docs/SOURCE_RECOVERY_20261102.md)区分历史运行和本次检查。

代码学习从[完整方法与调用路径](docs/source-reading/METHOD.md)开始，再打开[源码逐行阅读器](docs/source-reading/READING.html)或[Markdown文件目录](docs/source-reading/README.md)。[数学推导与边界](docs/source-reading/THEORY.md)解释默认共享读出的保序性质；[当前研究方案](refine-logs/theory-grounding-20261102/FINAL_PROPOSAL.md)记录独立审查后的修订、用户确认的无监督约束和待验证事项。

**默认方法已从旧 R4 改为 `shared`。** 复现原行为时必须显式传入 `method="r4"` 或 `--method r4`。默认改变沿用远端接口状态，不代表共享方法已全面优于 R4；三任务 R4 参考数组保留原始字节。

## 安装与检测

Python 3.11+，建议使用独立虚拟环境。依赖集中在 `pyproject.toml`。

```bash
python -m pip install -e '.[experiments,dev]'
token-detect --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --task QA --input examples/request.json --output result.json
```

```python
from myproject import HallucinationDetector

detector = HallucinationDetector.from_pretrained(
    "/path/to/Meta-Llama-3.1-8B-Instruct", task="QA", method="shared",
)
source = "The museum opened in 1998."
tokens = detector.predict(
    prompt=f"Passage: {source}\nWhen did the museum open?",
    source=source,
    response="The museum opened in 1997.",
)
```

`source` 必须在完整提示中出现且仅出现一次。结果包含 `token_id/text/start/end/score`；字符位置对应原回答。高分表示来源风险，不是校准的事实错误概率。库调用静默、在内存运行，不写中间文件。支持 QA、Summary、Data2txt；观察模型和数据另行提供。

| method | 用途 |
| --- | --- |
| shared | 默认固定 Beta 共享读出，无需 FIT |
| persistent / witness | 联合延续与混合观测研究候选 |
| responsibility | 个体专家责任诊断 |
| r4 | 原任务配方与随包分发的冻结参考 |

其他方法见 `token-detect --help`。[主库说明](myproject/README.md)解释接口和分数边界。

## 数据集与实验

```bash
token-experiment prepare-dataset --name ragtruth --data /path/to/RAGTruth/dataset \
  --split test --task QA --output outputs/qa/inputs
token-experiment predict-dataset --input outputs/qa/inputs \
  --model /path/to/Meta-Llama-3.1-8B-Instruct --task QA \
  --method shared --output outputs/qa/scores
token-experiment evaluate-dataset --scores outputs/qa/scores \
  --label-granularity token --output outputs/qa/evaluation
```

准备、观测、评分与评价分开；推断不读取标注，封存分数后才评价。断点续跑绑定输入、模型、代码与方法身份，每个输出目录只允许一个写进程。

`r4-detect`、`r4-experiment` 保留为当前入口的命令行别名，命令名不会强制选择 R4。旧 Python 包 `myproject_experiment` 已迁到 `experiments`，导入路径需更新。

- [实验入口](experiments/README.md)
- [远端实验账本](experiments/EXPERIMENTS.md)
- [最新研究结论与局限](docs/LATEST_RESEARCH.md)

RAGTruth 人工字符跨度、HalluRAG 句标签投影和金融 silver 标签有不同证据范围。完整研究稿、内部审查和工程检查不构成新检测器突破或录用证明。

## 验证

```bash
python -m pytest -q
python -m pip wheel --no-deps --wheel-dir dist .
r4-experiment verify --packs /path/to/historical/packs \
  --baseline /path/to/r04_tempered_continuity_v1 --output outputs/verification.json
```

`verify` 需要外部历史制品，只核对冻结参考、token 身份与分数；不会产生新 AUROC。CI 从仓库外测试已安装的 wheel。远端完整测试与原始运行产物未全部恢复，本次测试不冒称远端完整测试重跑。原始数据、模型、私人会话和运行输出不随源码发布。代码许可证尚未指定。
