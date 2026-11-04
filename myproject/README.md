# V3 主线方法

入口是 `run.py:main()`，主类是 `api.py:HallucinationDetector`。调用顺序是文本编码 → 两种来源条件前向 → 原 token 对齐 → 返回观测；调用者可以保存观测，再单独评分。

```python
from myproject import HallucinationDetector, Observations

detector = HallucinationDetector.from_pretrained("/path/to/model", device="cuda:0")
observed = detector.extract(
    prompt="Passage: The museum opened in 1998. When did it open?",
    source="The museum opened in 1998.",
    response="It opened in 1997.",
)
observed.save("observations.npz")

restored = Observations.load("observations.npz")
tokens = HallucinationDetector.score(restored)
```

`extract(prompt, response, source)` 返回内存中的 `Observations`，不写文件。`score(observed)` 是静态方法，不需要模型实例、权重或 GPU。`predict(prompt, response, source)` 依次调用提取与评分，直接返回结果。

## 输入、数据与分数

非空 `source` 必须是 `prompt` 中唯一出现的子串。`extract()` 要求回答至少包含一个 token；`predict()` 对空回答返回空列表。回答在两种条件下使用同一组原 token 和已有历史，不重新生成。

`Observations.pack` 包含五个同轴数组：

| 字段 | 含义 |
| --- | --- |
| `full_gap` | 无来源 logp 减有来源 logp |
| `target` | 有效 token 在原回答 token 序列中的位置 |
| `token_id` | 对应的原 token ID |
| `answer_index` | 回答编号，单条提取默认为 0 |
| `unit_index` | 原标点/换行分段编号，长段每 64 token 切分 |

`response` 保存原回答文本、完整 token IDs、字符 offsets、特殊 token IDs 和分段。保存为一个压缩 NPZ，元数据编码成 JSON，读取时禁用 pickle。特殊 token 和空字符区间不进入有效评分轴，其余 token 保留原位置。

结果每项为 `token_id/text/start/end/score`，字符位置索引原回答。高分表示较高来源风险，分数是模型内两种状态的对数优势，不是校准的事实错误概率。片段内推断使用后续 token 的观测，因此属于离线回答检测。

## 文件与计算路径

| 文件 | 职责 |
| --- | --- |
| `run.py` | 解析参数、调用提取/评分、保存观测或结果 |
| `api.py` | `HallucinationDetector.extract/score/predict` 主流程 |
| `inputs.py` | prompt/source 编码、删除来源 token、回答编码和分段 |
| `model.py` | 本地模型加载、KV prefill、分块前向、原 token 的 logp |
| `measure.py` | 原 token 轴、字符结果、分段边界与浮点顺序 |
| `observations.py` | `Observations` 数据对象及 NPZ 保存/读取 |
| `witness.py` | `WitnessDetector` 和 V3 两状态推断 |
| `__init__.py` / `__main__.py` | 公共导出 / Python 模块入口 |

模型固定已有回答，逐 token 计算：

```text
full_gap = log p(token | 无来源, 原回答历史)
         - log p(token | 有来源, 原回答历史)
```

来源删除移除与指定来源字符范围重叠的 prompt token，保留其他提示内容和原回答历史。V3 将每个 gap 转为 `logaddexp(0, full_gap) - log(2)`，对应把有来源与无来源两个预测分布等权混合后，与有来源分布作比较。然后在每个原片段内执行两状态前后向推断，对状态比例与相邻状态复制参数的固定均匀先验积分；无需 FIT 或任务标签。

为保持历史数值，来源存在时沿用 eager 与 `log_softmax`，来源删除时沿用 SDPA 与 `gather - logsumexp`，并保留片段内的浮点加减顺序。主线提取只计算完整历史的来源差，不计算 local/route 等研究观测。实验中的 V3 和消融调用同一组 `witness_emissions`、`transition_grid`、`state_marginals` 函数。

## 命令行与迁移

```bash
token-detect extract --model /path/to/model --input examples/request.json --output observations.npz
token-detect score --input observations.npz --output scores.json
token-detect predict --model /path/to/model --input examples/request.json --output scores.json
```

`python -m myproject` 提供相同子命令。`extract` 必须指定输出路径；`score/predict` 省略 `--output` 时输出 JSON。模型路径与设备显式传给 `from_pretrained()`；主线无方法选择、任务配方或参考分布参数。

原多方法接口已迁至 `experiments.inference.HallucinationDetector`，原文本 CLI 是 `r4-detect`。R4、V1/V2、其他候选、消融与评价见 [实验层说明](../experiments/README.md)。这次接口迁移不保留旧模块别名；主库不导入实验包。
