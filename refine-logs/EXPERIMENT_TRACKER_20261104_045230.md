# 幻觉检测执行 Tracker：显式模型续跑已启动

**快照：2026-11-04 04:52:30，Asia/Shanghai。** 兼容计划验证完成，新的SCORE续跑任务已真实启动。当前为**RUNNING，未完成、未最终审计、未评价**；目标标注值尚未读取。[上一版本](EXPERIMENT_TRACKER_20261104_044822.md)及此前历史保留。

| 实际任务 | 身份 | 状态 | 已发生的事实与边界 |
| --- | --- | --- | --- |
| M0–M4历史机制/仪器链 | 见[04:10历史tracker](EXPERIMENT_TRACKER_20261104_041029.md)与[计划](EXPERIMENT_PLAN.md) | DONE / 原边界保留 | M3所测K路径负结果、M4单例描述与严格FP64 inconclusive不改写、不重跑 |
| M5五来源有限对照 | f47b032f首轮、7cf99af7续跑 | DONE / supported_scope | 总5model/5head=2组复用+3组新算；三项固定对照与独立审计通过，约3GiB显存、无OOM；不是自然AUROC或新检测器 |
| LOCAL DeBERTa七文件/原始900规划 | inventory8bef1cfa…、plan b1c3c43b… | DONE / pinned and sealed |900答/150来源/8240既有完整句/8247旧测量unit/178343原targets；K18162、planning unknown0、最长512、NO dedup |
| 原SCORE v1 |8902c7819378d476ac0c2a78e31eb8bbd4d6284598ae7c31ddbbe8d4dc623a4f | STOPPED / pre-forward resource guard |801完整预测、7344完整unit与第802例部分成功raw已封存；16160成功模型输出含2canary；第16161接口尝试在模型前被512MiB资源门拒绝，没有模型OOM |
| 兼容计划v2 |caba7e54f826a447875473da20a5d9b0036287a3b42b680f4dde09853a4e0178 | **DONE / supported_scope / 0 MODEL** | exit0，注册finish完成；seal `6c9441f27a013f98b1f65a59d09284f4c503f45426912971d1cdb8cdded34167`。旧stage AST、140依赖/运行时、raw前缀/成本通过；units/protocol/modelidentity保持原字节，仅派生新source/续接证明 |
| SCORE显式续跑v2 |c6b4520fb08d4fcf70988292c2e88a05b5f10b83f3b5fb0433e5d0522e863dcb | **ACTUAL RUNNING** |04:52:30真实launch，launcher PID25604。先校验旧archive和canonical前缀，再复用16160个成功结果，仅余2003新模型调用预算；不再执行原801例的成功模型请求 |
| 完整900独立评分审计 | 待v2完整seals与sidecar | WAITING | 先审完整原轴/窗口/风险、801完整预测byte一致与partial802 raw保留、code/source/model及恢复物理计数；不执行模型 |
| 主线最终评价/R4-V3表 | evaluate_dataset，2000bootstrap/seed17/native | WAITING | 独立审计完成后才打开标注值，报告完整178343轴AUROC/AP、top5% FN/FP、coverage/unknown和费用；复用原封存R4/V3 |
| 新主线无监督检测器 | 无新方法准入 | **NOT_ADMITTED** | 固定外部监督DeBERTa仅为比较系统；M5有限source变体margin仍不是单输入原R风险。六任务各0.8目标未完成 |

## 固定恢复合同与实际计数

原失败root封存SHA仍为`55d00ac080e61c465139d63570bd35e6fc5c070ea43645a5c40be4b939cbbe3a`。原有效raw逐canonical请求/顺序/SHA复用，包含第802例中已成功但尚未形成完整预测的部分；保留原失败和所有有效证据，不覆盖历史。

**预算和已完成值分开：** 16160是旧run已成功的物理模型结果；2003是本次新物理模型调用预算，尚不宣称全部完成。只有本次完全成功时，合并有效模型结果才为18163，物理接口请求总数为原16161（含guard1）+新2003=18164。原predict及stdout中的attempt/cost计数包含cached replay，不能当作本进程新模型调用数。`<new-score-output>.recovery.json`在结束时单列真实新调用、复用结果、旧guard及output/compatible-plan seals。

原2canary作为旧资格结果复用；本次不额外推理canary，也不宣称fresh-resident canary已测。恢复仅增加显式编排和证据复用，原predict/helpers/HF数值核心保持；29项相关测试和独立审查已通过，这些工程结果不代表自然检测成绩。

## 不变的评分与停止门

原S完整token窗口、完整既有R句子、FP32 GPU/batch1、512含special、`1−max_chunk P_entail`、neutral非entailment及预定unknown0.5不变。完整178343 target/token/offset唯一membership与8240句子scope不变；不新分句、LLM改写、调阈值或融合R4/V3。

512MiB host/GPU floor与6GiB allocator cap保持原合同。若新资源、数值或身份门失败，继续保留并停止，不自动重试、降门限/精度、缩窗口或把剩余样本填unknown冒充完成。本次为独立审查、新内容指纹claim下的显式单次续跑，`automatic_retry=False`不变。

## 固定下一步

1. 等v2完成并封存全部900分数、原轴、unknown及外部恢复sidecar，核清实际成功模型与接口拒绝计数。
2. 做独立nonlabel code/source/raw/score审计：原18162逻辑pairs、全部16160旧raw与partial802复用、2003新模型结果、原801预测及完成unit字节、140依赖/核心AST/运行时、完整178343投影。strict辅助数值检查若inconclusive保留，不改界救结果。
3. 审计通过后才由既有`experiments.evaluate.evaluate_dataset`打开原标注值；标注路径/SHA、2000/17/native已冻结。严格join900 roster/response SHA，并在同原轴上对照封存R4/V3的AUROC/AP、预算FN/FP、coverage和实际费用。审计或bootstrap不触发重新模型评分。

Windows小文件resolve/hash与CPU回放/I/O是真实工程费用，不是新模型推理，也不免费扣除。当前没有自然AUROC、无误差结论或新无监督模块晋升；MiniCheck下载/网络重试仍停止。
