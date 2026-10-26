# R4 token detector

逐 token 的 RAG 幻觉风险检测，输入为完整提示、其中的实际检索文本和已有回答。
GitHub：[DwightEd/token_detector](https://github.com/DwightEd/token_detector)。

当前维护的是冻结 R4 基线。研究候选及其探索性成绩未作为新方法并入。
旧 R4 的任务配方曾用带标签 TRAIN-DEV 选型，参考分布拟合不使用标签；不能将整个研发过程称为从未使用标签。

## 安装

Python 3.11+，建议使用独立虚拟环境。在项目根目录执行：

```bash
python -m pip install -e .                     # 检测接口
python -m pip install -e '.[experiments,dev]'   # 数据准备、评价及测试
```

依赖版本集中在 `pyproject.toml`。`requirements.txt` 是完整开发安装的快捷入口。
观察模型使用本地 Meta-Llama-3.1-8B-Instruct；模型权重和数据集另行提供。
冻结参考针对该观察模型，换模型需用无标签 FIT 样本重新校准。

## 单条检测

```bash
r4-detect --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --task QA --input examples/request.json --output result.json
```

也可使用 `python -m myproject`，或在 Python 中调用：

```python
from myproject import HallucinationDetector

model = HallucinationDetector.from_pretrained('/path/to/Meta-Llama-3.1-8B-Instruct', task='QA')
source = 'The museum opened in 1998.'
tokens = model.predict(
    prompt=f'Passage: {source}\nWhen did the museum open?',
    source=source,
    response='The museum opened in 1997.',
)
```

`source` 必须在完整提示中出现且仅出现一次。每条返回 `token_id/text/start/end/score`；
字符位置对应原回答，分数越大风险越高。库调用在内存中完成，不保存中间文件。
支持任务为 `QA`、`Summary`、`Data2txt`。完整计算与参数见 [主线说明](myproject/README.md)。

## 数据集流程

```bash
r4-experiment prepare-dataset --name ragtruth --data /path/to/RAGTruth/dataset \
  --split test --task QA --output outputs/ragtruth-qa/inputs
r4-experiment predict-dataset --input outputs/ragtruth-qa/inputs \
  --model /path/to/Meta-Llama-3.1-8B-Instruct --task QA \
  --output outputs/ragtruth-qa/scores
r4-experiment evaluate-dataset --scores outputs/ragtruth-qa/scores \
  --label-granularity token --output outputs/ragtruth-qa/evaluation
```

HalluRAG 的官方 pickle 需先用 `r4-experiment export-hallurag` 导出为 JSONL，命令见实验说明。

标准适配输出为 `inputs.jsonl`、`annotations.jsonl`、`manifest.json`；推断只读取输入，
封存分数后评价才读取标注。支持断点续跑；输入、模型、参考或代码变化会拒绝续跑。
每个输出目录只允许一个写进程。新增数据集实现 `datasets.py` 中显式选择的适配函数，
返回 `Sample` 和独立标注，继续调用相同检测接口。

- **RAGTruth**：原生错误字符区间映射至实际 token。
- **HalluRAG**：读取生成器实际收到的检索块；句标签扩展到 token 是伪标签评价，不能等同于词内精细定位。
- **ClaimDB**：可以整理原生 claim 与数据库路径；当前文本检测器不支持 SQLite 证据，也不提供其 token 检测结果。

原始文件格式、未知标签、来源分组 bootstrap 与历史缓存命令见 [实验说明](myproject_experiment/README.md)。

## 结构

```text
myproject/             # 稳定检测接口与算法；references/ 随安装包分发
myproject_experiment/  # 数据适配、测量缓存、评价；单向依赖主线
examples/              # 最小输入示例
tests/                 # 可移植的 CPU 回归测试，含微型随机 Llama
tools/verify_frozen.py # 对外部历史制品逐位复验，无硬编码服务器路径
```

保留现有两个包名和兄弟结构，避免改动导入与历史实验入口。
`ASSEMBLY_MANIFEST.json` 保留最初复制时的文件哈希和静态检查记录，非当前构建验证报告。
本仓库不携带模型、原始数据、缓存和历史实验结果；相关目录已加入 `.gitignore`。

## 验证

```bash
python -m pytest -q
python -m pip wheel --no-deps --wheel-dir dist .
python tools/verify_frozen.py --packs /path/to/historical/packs \
  --baseline /path/to/r04_tempered_continuity_v1 --output outputs/verification.json
```

冻结复验只检查参考数组、DEV/TEST 原 token 轴与分数数值，不读取真假标签，不产生新 AUROC。
工程验证的实测范围见 [验证记录](docs/VALIDATION.md)。GitHub Actions 配置提供 CPU 检查。
数据与模型的获取、许可遵循各自来源；当前项目尚未指定代码许可证。
