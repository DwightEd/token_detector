> 最新研究状态与全部真实结果：[三轮迭代报告](docs/ITERATION_REPORT_20261029.md)。V3比较基线保留，三个新候选未晋升；下一目标契约已有审查，尚非有效新检测器。

# Token detector

无监督逐token的RAG来源风险检测。主线是固定Beta(1,1)的共享状态模型：
预先确定的文本区间限定联合分析范围，每个token保留自己的来源对照因子，不将区间分数广播成所有token的判据。
证据支持包含材料直接支持及有依据的前序步骤传递，不能等同于字词逐字出现。
来源风险分数仍不保证逻辑正确；步骤依据观测是持续研究目标。

本仓库是唯一主工程：[DwightEd/token_detector](https://github.com/DwightEd/token_detector)。
开始先读[当前交接](docs/HANDOFF.md)、[当前计划](experiments/PLAN.md)和[运行指导](docs/RESEARCH.md)。
旧实验只查[唯一账本](experiments/EXPERIMENTS.md)，不将过期计划当下一步。

## 安装与单条检测

Python 3.11+；依赖只在pyproject.toml声明，SciPy随共享主线安装。数据和观察模型另行提供。

```bash
python -m pip install -e .
python -m pip install -e '.[experiments,dev]'
token-detect --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --task QA --input examples/request.json --output result.json
```

```python
from myproject import HallucinationDetector

detector = HallucinationDetector.from_pretrained(
    "/path/to/Meta-Llama-3.1-8B-Instruct", task="QA",
)  # 默认shared；无需FIT或参考文件
source = "The museum opened in 1998."
tokens = detector.predict(
    prompt=f"Passage: {source}\nWhen did the museum open?",
    source=source,
    response="The museum opened in 1997.",
)
```

source须在完整prompt中出现一次。返回原回答的token_id/text/start/end/score；
分数越高风险越高，是专家责任log odds，不是校准的事实错误概率。
库调用静默、在内存运行，不保存中间文件。三任务使用相同full-view固定先验读出。
完整接口与公式见[主线说明](myproject/README.md)。

V1、V2、V3现在都在主库中，由同一个接口选择；默认仍为shared，方法接入不代表默认晋升。

| `method` | 读出 | 状态 |
|---|---|---|
| `shared` | V1固定Beta共享 | 原默认 |
| `persistent` | V2共同倾向＋局部延续 | 已有完整结果 |
| `witness` | V3混合观测＋联合延续 | 最新已验证候选 |
| `responsibility` | 同一V3模型内的个体专家责任 | 固定研究诊断，不能解释为事实概率 |
| `r4` | 原任务配方＋外部无标签参考分布 | 冻结强对照 |

```python
from myproject import HallucinationDetector, SharedDetector

detector = HallucinationDetector.from_pretrained("/path/to/model", method="witness")
tokens = detector.predict(prompt=prompt, source=source, response=response)

# 缓存观测也调用同一个主线对象，不在实验目录重写评分。
scores = SharedDetector(method="witness").score(pack)
```

对应命令是 `token-detect --method witness ...` 和 `token-experiment predict-dataset --method witness ...`。
[代码地图](docs/CODE_MAP.md)说明每个文件的职责和实际调用链；[V3逐步说明](docs/METHOD_V3.md)解释算法。

## 数据集统一流程

```bash
token-experiment prepare-dataset --name ragtruth --data /path/to/RAGTruth/dataset \
  --split test --task QA --output outputs/qa/inputs
token-experiment predict-dataset --input outputs/qa/inputs \
  --model /path/to/Meta-Llama-3.1-8B-Instruct --task QA --method witness --output outputs/qa/scores
token-experiment evaluate-dataset --scores outputs/qa/scores \
  --label-granularity token --output outputs/qa/evaluation
```

默认shared预测不读FIT或R4参考。要缓存原生观测，可用transfer collect（--fit可省略）→
transfer score（默认shared）→transfer evaluate。输入/源码/方法变化会拒绝在原目录续跑。
所有分数先封存，再读取原标注。完整命令见[实验说明](experiments/README.md)。

- RAGTruth：原生字符错误区间映射至实际token。
- HalluRAG：使用生成器实际收到的材料；句标签扩至token是伪标签，不能冒充词内定位。
- FRED：金融FinQA/TATQA合成silver编辑区间；原问题未发布，source-only迁移条件明确记录。
- ClaimDB：可整理原生claim/数据库；当前文本检测器不支持SQLite来源或token成绩。

## 当前目标与证据

完整V3结果：FinQA AUROC/AP=.802656/.325033，TATQA=.842992/.465722；点值目标已达到。
完整RAGTruth QA=.846975/.214833、Summary=.762337/.104562、Data2txt=.755716/.145209；QA/Data2txt仍低于R4。
这些TEST均已参与开发，不是未见泛化或顶会就绪结论。
FRED金融为silver编辑标签，RAGTruth为人工错误跨度；两金融绝对AUROC区间仍跨.8。
新`responsibility`诊断已完整测试，五任务AUROC/AP均低于V3，不晋升；
它恢复了monthly并减少一例正确计算误报，整体代价仍明显，不能只报告这些局部改善。
[完整当前结果](docs/RESULTS_CURRENT.md)保留所有对照、误报漏报、评审结论及数据资格限制。

## 历史R4对照

```bash
token-detect --method r4 --model /path/to/model --task QA --input request.json
```

Python传method="r4"；明确提供旧R4 references也选择该对照。
其任务配方经过带标签DEV选型，CDF FIT本身无标签，不将整个研发过程称从未看标签。
旧r4-detect/r4-experiment命令和分发名因已有调用保留，共用实现；方法默认已变为shared。

## 目录与验证

myproject/维护主库与唯一V1/V2/V3/R4推断，experiments/维护数据/缓存/评价，
experiments/research/放确有差异的候选与调用主库的消融，tests/做回归，outputs/<run-id>/保存生成物并忽略Git。
不恢复已删除的experiment_results，不在结果目录复制算法。

```bash
python -m pytest -q
python -m pip wheel --no-deps --wheel-dir dist .
```

构建前清generated build缓存；仓库外安装测试并检查源码/资源集合。
实际验证范围与旧版本重放见[验证记录](docs/VALIDATION.md)。
