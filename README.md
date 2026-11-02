# Token detector

最新优先级是[逐项查明 FN/FP 的错误机制](docs/source-reading/FNFP_MECHANISM.md)，并[核对原始来源、回答及模块分数](docs/source-reading/FNFP_CASES.md)。完整缓存已恢复。固定等权对照未达标，保留为失败实验记录；当前算法默认值未更改。


逐 token 的 RAG 来源风险检测与研究工具。输入为完整提示、其中实际出现的来源文本和已有回答，返回原 token 的位置与分数。

研究主线固定为**无监督检测，真值仅用于冻结后的最终评价**，不用于训练、调参、选方法或风险校准。完整约束见[用户确认记录](refine-logs/theory-grounding-20261102/USER_CONSTRAINTS.md)。

最新任务是[逐任务AUROC≥0.8：R4/V3统计、真实FN/FP与模块优化](docs/source-reading/PERFORMANCE_REVIEW.md)。允许多模块组合，不要求一个统一概率模型。按用户要求，HalluRAG含答案891答为主要对照；已恢复R4在该组AUROC/AP **0.8071/0.5188**，V3 **0.7669/0.4561**。完整分层与采集差异随表说明，旧VAL132结果不再与完整TEST混比。现成标注数据可以直接用于评价。

**当前保留的金融研究候选是 V3 / witness。** 先看[无标注识别边界与当前决策](docs/source-reading/UNLABELLED_LIMITS.md)、[V3优化](docs/source-reading/METHOD_V3.md)、[六组完整对照](docs/RESULTS_CURRENT.md)与[当前诊断计划](refine-logs/EXPERIMENT_PLAN.md)。[此前关系组合](docs/source-reading/INTEGRATED_DESIGN.md)因自然语义确认前提未解而退出默认主线，保留为受限可选分支。离线阅读从[识别边界页面](docs/source-reading/UNLABELLED_LIMITS.html)进入；V3并非所有任务/观察模型最优，没有新检测结果。

浏览器阅读从[公式与机制交互指南](docs/source-reading/VISUAL_GUIDE.html)进入；[完整公式](docs/source-reading/THEORY.html)与[论文主张](docs/source-reading/PAPER_STORY.html)均可离线排版。下载仓库后打开HTML；GitHub在线查看可读[文字版](docs/source-reading/PAPER_STORY.md)。

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
