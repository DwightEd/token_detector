# myproject

R4 幻觉风险检测器：来源效应、逐词创新与有界全文持续。

主线直接接收完整 prompt、其中的来源文本和已生成回答，在内存中计算，
只返回逐 token 的最终结果。调用不打印阶段日志，不保存中间特征。

## 直接调用

在仓库根目录执行 `python -m pip install -e .`；依赖见根目录 `pyproject.toml`。支持本地
Meta-Llama-3.1-8B-Instruct；三任务的已校准参考参数随包放在 `references/`。

```python
from myproject import HallucinationDetector

detector = HallucinationDetector.from_pretrained(
    "/path/to/Meta-Llama-3.1-8B-Instruct",
    task="QA",  # QA / Summary / Data2txt
    device="cuda:0",
)

source = "The museum opened in 1998."
result = detector.predict(
    prompt=f"Answer using the passage.\nPassage: {source}\nWhen did it open?",
    source=source,
    response="It opened in 1997.",
)
```

`source` 是完整 prompt 中出现一次的非空来源文本，表示待测量的整个来源区域，
不需要提供正确实体、正确关系或错误位置。
`result` 是列表，每个元素只有 `token_id`、`text`、`start`、`end`、`score`。
字符位置对应输入的原回答；不输出特殊 token 或空字符位置。
分数越大表示风险越高，不是校准后的错误概率。

这是离线检测，允许回答的后续位置参与最终推断。

## 对照代码的算法流程

| 步骤 | 实际计算 | 代码 |
|---|---|---|
| 1 编码 | 保留完整 prompt 和原回答的 token 与字符位置；按换行、句末标点形成 local 单元，最长 64 token | `inputs.py` |
| 2 测量来源作用 | 冻结模型，对实际生成词测来源保留／删除与全文／单元内历史四种 logP；计算删源减留源的差值 | `observer.py`、`source.py`、`model.py` |
| 3 测量路由 | 用 attention 与物理头的 OV 消息范数计算历史／来源相对作用，再取原位置 16 token 窗口 | `routing.py`、`measure.py` |
| 4 计算来源先验 | 在每个单元内对来源差值做温度 log-mean-exp，用已校准的分位数参考映射到秩 | `detector.py:tempered_prior`、`quantile_rank` |
| 5 保留逐词差异 | 对原位置小窗口的来源差值取参考秩，再减去单元内均值，得到每词创新量 | `detector.py:target_window_mean`、`center_units` |
| 6 合成风险 | 来源先验、route 秩和逐词创新按固定任务参数合成 unary | `detector.py:Detector.score` |
| 7 全文持续 | 在同一回答的相邻有效 token 间做精确前后向消息推断，间隔越大连接越弱，返回每词最终分数 | `graph.py:smooth_scores` |

来源差值为 `delta = logP_without_source - logP_with_source`。
正值表示来源压制实际生成词；来源促进某个词，不自动表示事实支持。

`unary = (1-route_weight)*prior + route_weight*route_rank + innovation_weight*innovation`。

| 任务 | 来源差值视图 | 温度 | 创新半径 | 创新权重 | route 权重 |
|---|---|---:|---:|---:|---:|
| QA | local/full 均值 | .1 | 2 | .1 | .25 |
| Summary | full | .25 | 4 | .25 | 0 |
| Data2txt | local/full 均值 | .1 | 4 | .25 | 0 |

链的 emission 是 `2*(unary-.5)`，相邻连接 penalty 为 `.5/原位置间隔`。
每词有独立分数，单元共同先验不等于整段同分。这是相邻持续链，不是完整多跳溯源图。

这些计算中间量只在内存中使用。默认调用不创建特征文件、运行目录或评价报告。

## 参考参数

`from_pretrained` 默认加载包内当前 R4 对应任务的参考参数，无需重新执行实验。
旧 R4 的任务配方曾根据带标签 TRAIN-DEV 的 AUROC 选型；无标签仅指 FIT 参考估计，
不能将整个研发过程称为从未使用标签。跨数据集直接复用这些参考属于迁移测试。

换观察模型或校准数据时，可用无标签样本重新校准：

```python
detector.fit(fit_examples)  # 每条只有 prompt、source、response
# 仅在显式请求时保存参数。
detector.save("references.npz")

other = HallucinationDetector.from_pretrained(
    "/path/to/Meta-Llama-3.1-8B-Instruct",
    task="QA",
    references="references.npz",
)
```

低层数值接口为 `myproject.detector.Detector` 的 `fit(pack)` 与 `score(pack)`；
后者只返回最终分数数组。高层入口位于 `api.py`，不要求调用者构造这些中间量。

## 命令行

`request.json` 包含 `prompt`、`source`、`response` 三个字符串。

```bash
python -m myproject --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --task QA --input request.json --output result.json
```

只写最终 token 结果；省略 `--output` 时，最终 JSON 输出到标准输出。

## 实验工具

数据集准备、缓存读取与导出、评价、诊断和测试位于独立的
[`myproject_experiment`](../myproject_experiment/README.md)。实验工具调用主线算法，
主线不依赖实验目录。

新数据的处理入口见实验工具的 `prepare-dataset → predict-dataset → evaluate-dataset`。
适配后仍调用这里的三个字符串接口，返回相同 token 轴和分数；原生标签只进入独立评价。
当前检测要求文本来源，数据库或其他模态须先定义相应的证据方法。
