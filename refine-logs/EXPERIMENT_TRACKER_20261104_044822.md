# 幻觉检测执行 Tracker：资源门停止后的显式续接

**快照：2026-11-04 04:48:22，Asia/Shanghai。** 原NLI评分在801/900后因冻结资源门拒绝下一次调用而停止；成功原始结果完整保留。当前仅兼容计划验证job已真实启动，**0模型调用**；新模型评分尚未启动。目标标注值未读取，自然AUROC/AP仍未计算。

[上一实际快照](EXPERIMENT_TRACKER_20261104_041029.md)与[实验计划](EXPERIMENT_PLAN.md)保留；此次只更新执行状态，不改变模型、精度、窗口、风险、原评价轴或资源门限。

| 实际任务 | 身份 | 状态 | 已发生的事实/下一门 |
| --- | --- | --- | --- |
| M0–M4机制与仪器记录 | 见04:10历史tracker与结果页 | DONE / 边界保留 | M3所测K路径负结果、M4单例描述及严格FP64 inconclusive不改写、不重复 |
| M5五来源有限范围对照 | f47b032f首轮、7cf99af7续跑 | DONE / supported_scope | 共5model/5head=2组复用+3组新算，三项固定对照及独立审计通过；约3GiB显存、无OOM，不是自然AUROC或新检测器 |
| LOCAL固定DeBERTa七文件 | inventory8bef1cfa… | DONE / pinned | 不依赖下载；外部MNLI/FEVER/ANLI监督比较系统，不是新无监督方法 |
| Data2txt900原始完整规划 | b1c3c43bc91087ade76e011d4d715030913e357c889729a750a3abd472aa165c | DONE / sealed |900答、150来源、8240完整既有句、8247旧测量unit、178343原targets；K18162、unknown0、最长512，NO dedup |
| 原始NLI全评分v1 |8902c7819378d476ac0c2a78e31eb8bbd4d6284598ae7c31ddbbe8d4dc623a4f | **STOPPED / RESOURCE_GUARD** |801完整预测、7344完整句unit及第802例部分成功raw保留；16160次成功模型输出含2canary；接口尝试16161中的最后1次在模型前被512MiB门拒绝，不是模型OOM |
| 显式单次续接工程审查 | source3c4661d7… / peer reviewc537a61a… | DONE / engineering_only |29相关tests通过，原predict/helpers/HF数值核心保持；真实old1695 core AST核对通过。成功prefix按原canonical请求逐SHA/顺序复用，包括partial802，不能从801预测截断点重算 |
| 兼容计划v2验证/派生 |caba7e54f826a447875473da20a5d9b0036287a3b42b680f4dde09853a4e0178 | **RUNNING / 0 MODEL** |04:47真实launch；核旧140依赖、运行时、原raw前缀/成本、旧stage AST；原units/protocol/modelidentity字节保持，派生新source/provenance计划。Windows小文件resolve/hash为当前瓶颈，无GPU或网络任务 |
| 剩余NLI模型评分 | 尚无新SCORE claim/launch | **WAITING / NOT_LAUNCHED** | 兼容计划封存后，root另claim2003次新模型预算；前16160有效输出仅回放。原2canary资格复用，不宣称fresh-resident canary，不再次调用它们 |
| 完整900独立分数审计 | 未开始 | WAITING | 完整900/seals及恢复sidecar之后核原轴、全部风险、raw复用、source/code/model和实际物理计数；不调用模型 |
| 主线自然评价/R4-V3对照 | evaluate_dataset，2000/17/native | WAITING | 全分数封存与独立审计通过后才打开目标标注值；报告AUROC/AP、固定top5% FN/FP、coverage/unknown和费用 |
| 新主线无监督检测器 | 无新方法准入 | **NOT_ADMITTED** | M5有限source对照不是单次原输入风险；外部NLI比较器也不能改称新无监督创新。六任务各0.8目标未完成 |

## 原失败与续接计数

原失败root为`outputs/h017-relation-20261103/local-nli-datatxt900-score-v1`，封存SHA为`55d00ac080e61c465139d63570bd35e6fc5c070ea43645a5c40be4b939cbbe3a`。最后一次资源拒绝发生在模型前，原有效16160次输出和未完成第802例中的raw都必须复用；原801完整预测及历史失败记录不覆盖。不是从头重跑900，也不是只从802号预测文件边界继续丢掉部分raw。

在本次恢复完全成功的条件下，物理模型输出为16160旧+2003新=18163；物理分类接口请求总数为原16161（含guard1）+新2003=18164。原predict重建输出中的18163次attempt/cost字段是合并证据范围，包含cached replay，**不是新进程执行18163次模型**。单独`<new-score-output>.recovery.json`记录真实新调用、缓存复用和旧guard；stdout进度同样不能直接当新物理调用计数。

本次是独立审查和新内容指纹claim下的显式续接，`automatic_retry=False`不变。若新资源/数值/身份门失败，继续保留并停止，不自动重试、不降低512MiB host/GPU floor、不改6GiB allocator cap、FP32、batch1或512窗口，不将剩余样本填unknown伪造完成。当前不存在“恢复模型已经跑完”的结果。

## 信息、评价与费用边界

- 原S完整token窗口和完整既有R句子，风险保持`1−max_chunk P_entail`；neutral为非entailment，预定unknown0.5留在完整分母。planning unknown0不等于0错误、0FN或检测全对。
- 原178343 target/token/offset唯一membership及8240完整scope不变，不新分句、不LLM改写、不按标签挑模型/窗口/阈值，也不融合R4/V3分数。
- 标注路径与SHA已事先冻结并仅核存在/hash；最终评价仍为原`evaluate_dataset(..., repetitions=2000, seed=17, label_granularity='native')`，900 roster/response SHA由既有evaluator严格join。
- 模型峰值、真实wall时间和各过程费用以完整收据为准。原resource failure并不抹去已有计算费用；CPU复用/校验/I/O也不是免费。未完成时不发布partial AUROC，不因审计或bootstrap重评分。
- MiniCheck新下载/网络重试仍停止；恢复只使用已完整固定的LOCAL DeBERTa及原缓存。没有新的主方法准入、自然AUROC结论或目标标签值读取。

实际新SCORE只有在root给出真实claim和launch后才更新为RUNNING；本版本仅记录已发生的兼容计划job。
