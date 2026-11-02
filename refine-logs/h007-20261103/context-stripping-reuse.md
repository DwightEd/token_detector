# H007 后缀去除：精确缓存复用证明

结论：48答×6分配=288个 would-be target 请求，逐个完整 execution 字典都与现有零负担16答×6=96个独立请求完全相同；每个已有 key 被引用3次。不存在新模型执行，不能把删除回答后缀重新包装成新实验。

操作仅将原回答改为首个自动精确单元，保留其固定尾空格，并删除所有后续字符。来源、原 prompt、来源占位转换后的 context、标签分配、model identity、protocol/source hash、scoring 和 admission 字段都保持；任何一项不等都会立即失败。实际使用不可修改109文件目标中的 focus.build_request/shared._context；助手未改 core/state。

风险只引用16条已封存 b0 parent completion 及其 child case 路径/哈希，没有重新聚合 logits、重算风险或读取正式 truth。48条逻辑实例仍全部列在 JSON 的288条映射中；该重复清单不是新独立观测。

直接引用 sealed summary 的 b0 数值：正确配对方向 5/8，平局 0；mean_wrong_minus_supported_risk=0.08139438195639939，AUROC=0.578125，AP=0.6897095959595959。whole 与 target 的这些既有值完全相同，去后缀不会改变它们，也不能修复最小关系条件已有的错误。

new_model_calls=0；posthoc_diagnostic=True；new_scores_or_metrics_computed=False。完整 input/design/observer/child-plan/summary/seal 绑定、96个对应缓存 key、16个风险引用与288条逐分配映射见 context-stripping-reuse.json。观察封存 SHA256：`36118e42cb86960ba5581a48739e00a0d9a344870586013205babbc8fa94bbf1`。
