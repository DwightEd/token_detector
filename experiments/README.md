# 实验层

本包管理数据准备、观测缓存、多方法比较、消融、验证、评价与汇报。候选方法及历史 R4 放在 `experiments/research/`，调用 `myproject/` 的编码、前向、原 token 轴和 V3 内核。主库不依赖实验层。

`token-experiment` 与 `python -m experiments` 是统一入口，`r4-experiment` 是其别名。各阶段的参数以 `--help` 为准。

## 数据集流程

```bash
token-experiment prepare-dataset --name ragtruth --data /path/to/RAGTruth/dataset \
  --split test --task QA --output outputs/qa/inputs
token-experiment predict-dataset --input outputs/qa/inputs \
  --model /path/to/model --task QA --method witness --output outputs/qa/scores
token-experiment evaluate-dataset --scores outputs/qa/scores \
  --label-granularity token --output outputs/qa/evaluation
```

准备阶段分离 inputs、annotations 和 manifest；推断只读输入，完成后封存原 token 分数；评价检查封存后才打开标注。断点续跑绑定输入、模型、方法和代码身份，修改它们应使用新的输出目录。

实验多方法入口保留历史默认 `shared`，V3 必须显式选择 `--method witness`。RAGTruth 人工字符跨度、HalluRAG 句标签投影、金融 silver 标签保持各自含义；不同观察模型与上下文合同的差别单列报告。

缓存流程使用 `transfer collect`、`transfer score`、`transfer evaluate`。主库保存的 `Observations` NPZ 只含 full gap、原 token 轴和回答元数据，不等同于历史实验包含 local/route 的 measurement pack；按对应入口读取，不改写旧 seal 或历史绝对路径。

## 多方法 API 与模块迁移

```python
from experiments.inference import HallucinationDetector

detector = HallucinationDetector.from_pretrained(
    "/path/to/model", task="QA", method="r4",
)
tokens = detector.predict(prompt=prompt, source=source, response=response)
```

原多方法 API 的 `method/task/references` 及 `fit/measure/measure_responses/save` 留在实验层。文本 CLI 使用：

```bash
r4-detect --model /path/to/model --task QA --method r4 \
  --input examples/request.json --output scores.json
```

`python -m experiments.detect` 提供相同接口；`token-detect` 只提供主库 V3 的 `extract/score/predict` 子命令。旧模块路径没有兼容转发。

| 迁移后的路径 | 职责 |
| --- | --- |
| `experiments/inference.py` | 原多方法文本 API |
| `experiments/detect.py` | 多方法文本 CLI |
| `experiments/research/r4.py` | R4 配方、参考分布与评分 |
| `experiments/research/readouts.py` | V1/V2 与候选读出组合 |
| `experiments/research/observation.py`、`source.py` | 历史 full/local、route 等观测 |
| `experiments/research/sequence.py`、`pooling.py`、`chain.py` | 研究状态模型、Beta 池化和 R4 链推断 |
| `experiments/research/references/*.npz` | 原 QA/Summary/Data2txt 冻结参考数组 |

V3 的状态积分只在 `myproject/witness.py` 实现；实验候选与消融复用该内核。R4 参考数组迁移保留原字节。R4 历史配方经过带标签开发集选型，无标签 FIT 仅指参考分布估计；保留它用于基线复现，不把历史选型包装成新的无监督方法。

## H017 与 NLI 对照

先读 [RESUME](../research-state/RESUME.md) 再继续研究。H017 的有限机制实验和完整 Data2txt900 NLI 比较均已完成，不重复启动已关闭的世界、头层或角色 QA 分支。

DeBERTa 是外部监督比较器，完整 Data2txt AUROC/AP 为 0.744306/0.097168，未优于冻结 R4/V3；它没有建立新无监督内部检测器。具体差异、置信区间、成本、错例与调用恢复见 [完整结果](../docs/results/h017_nli_detection.md)，机制证据见 [关系研究总结](../docs/results/h017_relation_binding.md)。

可选依赖为 `.[experiments,nli]`，入口是 `transfer predict-local-nli-dataset`。其规划、FP32/batch1 推断、512MiB host/GPU 保留门限及 SM120/Torch/CUDA 资格属于冻结运行合同；迁移设备需要另行资格验证。使用 `--prepare-only` 后再以 `--plan` 评分；已完成运行只复用，资源拒绝保留为未完成，恢复须有显式核验的历史请求前缀与来源兼容证明。标注由封存后的 `evaluate-dataset` 打开。

## 验证与证据

```bash
r4-experiment verify --packs /path/to/historical/packs \
  --baseline /path/to/r04_tempered_continuity_v1 --output outputs/verification.json
```

该 CPU 阶段需要外部历史制品，检查数值和 token 身份，不产生新 benchmark 成绩。`tools/verify_frozen.py` 调用同一实现。生成文件保存在 `outputs/<run-id>/`，历史正负结果保存在 [账本](EXPERIMENTS.md)；研究候选通过测试不等于提高了检测效果。
