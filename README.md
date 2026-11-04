# Token detector

对已有回答提取逐 token 的来源对比，并用 **V3 witness** 评分。主线在 `myproject/`，流程是编码文本、执行来源存在/删除两种条件的前向、对齐原 token、保存观测；评分可以单独读取保存的数据。

```bash
python -m pip install -e '.[experiments,dev]'
token-detect extract --model /path/to/model --input examples/request.json --output observations.npz
token-detect score --input observations.npz --output scores.json
```

输入 JSON 包含 `prompt`、`source`、`response`；非空 `source` 必须在完整 `prompt` 中出现且仅出现一次。模型和数据另行提供，Python 3.11+。

```python
from myproject import HallucinationDetector, Observations

detector = HallucinationDetector.from_pretrained("/path/to/model")
observed = detector.extract(
    prompt="Passage: The museum opened in 1998. When did it open?",
    source="The museum opened in 1998.",
    response="It opened in 1997.",
)
observed.save("observations.npz")
tokens = HallucinationDetector.score(Observations.load("observations.npz"))
```

`score()` 不加载模型或重新前向。需要一次返回结果时，使用 `detector.predict(...)` 或 `token-detect predict`。结果包含 `token_id/text/start/end/score`，位置对应原回答，高分表示来源风险，不是校准的事实错误概率。接口、九个 Python 文件及数据格式见 [主线说明](myproject/README.md)。

## 实验与接口迁移

多方法比较、消融、验证、评价与结果汇报放在 `experiments/`。数据集流程显式选择方法：

```bash
token-experiment prepare-dataset --name ragtruth --data /path/to/RAGTruth/dataset \
  --split test --task QA --output outputs/qa/inputs
token-experiment predict-dataset --input outputs/qa/inputs \
  --model /path/to/model --task QA --method witness --output outputs/qa/scores
token-experiment evaluate-dataset --scores outputs/qa/scores \
  --label-granularity token --output outputs/qa/evaluation
```

这是一次明确的接口迁移：主库只提供 V3 的 `extract/score/predict`，不再接收 `method/task/references`。原多方法 Python 接口位于 `experiments.inference.HallucinationDetector`；原多方法文本 CLI 使用 `r4-detect` 或 `python -m experiments.detect`。实验入口的历史默认仍是 `shared`，选择 V3 时传 `--method witness`。

R4 在 `experiments/research/r4.py`，其他读出在 `readouts.py`，三任务冻结参考数组在 `experiments/research/references/`。旧模块路径不提供转发别名；实验中的 V3 和消融复用 `myproject/witness.py` 的状态推断内核。详见 [实验层说明](experiments/README.md)。

本次将 V3 作为工程主线，保留 R4 与其他研究对照的数值和历史记录。重构不产生新的科学效果证据，也不表示 V3 已在所有任务上最优。旧 [逐行阅读材料](docs/source-reading/README.md)绑定重构前源码；当前调用路径以两份包 README 为准。

## 当前研究状态

[恢复状态](research-state/RESUME.md)记录当前可执行步骤；[H017 总结](docs/results/h017_relation_binding.md)保留关系错绑、时间单位和范围条件实验的实际正负结果。内部机制研究尚未形成新的已验证检测器，六任务各 AUROC ≥ 0.8 的目标未完成。

2026-11-04 完成的 Data2txt 对照保留全部 900 回答、150 来源、178343 个原 token：

| 方法 | AUROC | AP |
| --- | ---: | ---: |
| R4 冻结基线 | 0.798318 | 0.190392 |
| V3 / witness 冻结基线 | 0.755716 | 0.145209 |
| DeBERTa NLI 外部监督对照 | 0.744306 | 0.097168 |

NLI 仅新增这一任务的比较，不是新无监督内部方法。它的 AUROC 相对 R4 显著较低，相对 V3 的差值置信区间跨 0；完整方法、成本、错例和限制见 [检测评价](docs/results/h017_nli_detection.md)。研究方向仍允许超出 V3 的概率差框架，见 [关系研究合同](research-state/RELATION_BINDING_DIRECTION.md)。

研究约束是无监督检测，真值只在方法、参数和全部分数冻结后用于最终评价；历史已查看人群保持探索性。完整约束见 [用户确认记录](refine-logs/theory-grounding-20261102/USER_CONSTRAINTS.md)，历史实验见 [账本](experiments/EXPERIMENTS.md)。

## 数据与验证

下载位置、原始观测范围和换机步骤见 [远端数据位置与执行交接](REMOTE_EXPERIMENT_HANDOFF.md)。数据、权重、私人会话与运行输出不随 Git 发布；封存文件中的历史路径和原 seal 保持不变。

```bash
python -m pytest -q
python -m pip wheel --no-deps --wheel-dir dist .
r4-experiment verify --packs /path/to/historical/packs \
  --baseline /path/to/r04_tempered_continuity_v1 --output outputs/verification.json
```

CI 在仓库外测试已安装的 wheel。`verify` 需要外部历史制品，检查数值与 token 身份，不产生新 AUROC。最新 H017 验证收据见 [记录](docs/results/h017_nli_ci.json)；该收据属于重构前指定源码，不替代本次重构的测试。代码许可证尚未指定。
