# Callable token detector

`HallucinationDetector` 接收完整 prompt、其中唯一出现的 source 和已有 response，返回原 token 的 token_id/text/start/end/score。字符位置保留原回答中的空隙；库调用静默且不创建中间文件。

## 默认共享读出

```python
from myproject import HallucinationDetector

detector = HallucinationDetector.from_pretrained(
    "/path/to/Meta-Llama-3.1-8B-Instruct", task="QA", method="shared",
)
result = detector.predict(prompt=prompt, source=source, response=response)
```

每个 token 保留完整历史下的来源对照 `delta = logP_without_source - logP_with_source`。正值表示来源压制该 token，不等同于事实错误。

`pooling.cavity_scores` 对区间内的 Beta(1,1) 共享倾向精确积分，返回专家责任 log odds。自身对照没有先取区间均值；其他 token 会通过共享倾向影响分数。模型不保证正确的事实绑定。离线推断可使用回答的后续位置，不能当作在线提前告警。

原生观测集中在 observer/source/model；attention 与 segmentation 支持显式选择的分段；sequence 提供延续状态与诊断；SharedDetector 统一调用数值内核。

## 冻结 R4 对照

```python
legacy = HallucinationDetector.from_pretrained(
    "/path/to/Meta-Llama-3.1-8B-Instruct", task="QA", method="r4",
)
```

R4 使用随包分发的三任务 references/*.npz。其配方经过带标签开发集选型，无标签仅指 FIT 参考分布估计。换观察模型或参考人口时，可创建 `HallucinationDetector(model, tokenizer, method="r4")`，用无真值标签的例子调用 fit 并显式 save。共享方法不需要 FIT，调用其 fit 会报错。

显式传入旧 R4 references 也会选择 R4；共享方法不能同时使用 R4 参考。persistent、witness 等候选必须显式选择，接入不代表实证优越性成立。

## 命令行与实验

```bash
python -m myproject --model /path/to/model --method r4 \
  --task QA --input examples/request.json --output result.json
```

只写最终 token 结果；省略 --output 时输出 JSON。数据集、缓存、标注和评价由兄弟包 [experiments](../experiments/README.md) 管理，主库不导入实验包。
