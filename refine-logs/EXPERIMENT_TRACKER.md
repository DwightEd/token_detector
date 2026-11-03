# 关系错绑机制实验 Tracker

**2026-11-03实际更新（优先于下方历史PLAN）：** [结果与图表](../docs/results/h017_relation_binding.md)、[恢复记录](../research-state/RESUME.md)。新detector尚未准入，无新自然任务AUROC，六任务0.8未达成，当前无活动模型job。

| 实际阶段 | run（前8位；完整身份见research-state/runs） | 状态与边界 |
| --- | --- | --- |
| M0三任务source结构库存 | 80913833 | supported_scope；2700答/450来源、5806显式字段；散文语义unknown |
| M1 QA重放3+897 | 5f7cf0ad、48f5ce11 | supported_scope；900答资格通过，五坏实现各900拒绝；不是检测准确率 |
| M2原文/token对齐3+897 | 7d065e2e、8d6662cb | supported_scope；900答、124817原R token；语义所属未自动许可 |
| 原模型prefix与fullhead门 | 1fdf3766、7f222ddf | supported_scope；2 early-stop + 4 full model、3 head；不重复执行 |
| M3四来源Q/K/V干预 | 24dd6e0b | refuted，限所测K选择性最终介导；48 full model/44 head，V内容对照方向正；简例未复现高置信错绑 |
| M4自然时间观测（补充） | 31185253 | supported_scope，限1例描述；1 full model/1 head；ago低概率而首选BC，严格FP64附加界6/76 inconclusive保留 |
| 旧R4/V3读出链审计 | 复用旧封存结果，非新模型run | ago旧gap已告警，后续读出名次下降；旧BF16与新FP32跨运行桥未通过，分开解释 |
| 下方PLAN-RB-M4全人口检测评价 | 无 | 尚未准入/执行；不能将单例自然观测算为此阶段完成 |

累计55 original model（2 early-stop/53 full）、48 full-vocabulary head，0新目标标签/新检测分数。下一问题是关系特异读出及其合法改写、运算、条件范围反例，不能直接剪context或取max救单例。原源文件/已完成run/失败收据均保留。

---

下方为16:05未执行时的计划快照，不是当前运行状态。

时间：2026-11-03 16:05:41，Asia/Shanghai。对应 [实验计划](EXPERIMENT_PLAN_20261103_160541.md)。

**全部条目为PLANNED，未申领，不是实际run ID。** 新增模型调用、目标标签读取和新检测分数均为0；尚无新检测器准入或有效性结果。下载/哈希/header恢复的实际任务另有收据，不能算作这里的机制或检测成绩。固定别名和root state由主代理在设计审查后同步，本tracker仍不构成run claim。

观测范围为固定Llama3.1 observer对既有R的处理。其teacher-forced条件概率与路径干预不是不同原生成模型的真实采样概率或原始错误因果证据。

| 计划占位ID | 里程碑 | 目的/候选或对照 | 数据/范围 | 决定性指标或产物 | 优先级 | 状态 | 准入前置 |
|---|---|---|---|---|---|---|---|
| PLAN-RB-M0 | M0 | source-only类型事实与完整路径库存，不做新角色QA | 三任务原TEST各900及所有unknown | 原字节/offset、关系许可、重复值/范围/多跳/算术分层、source lineage seal | MUST | PLANNED / NOT_CLAIMED | 科学区别旧H004/H005/H012；无annotation输入 |
| PLAN-RB-M1 | M1 | 原Q/K重放资格：RoPE/GQA/因果/world掩码、FP16误差、原p身份 | 完成恢复的QA TEST900；先固定1–3 canary | local_attention/group_mass误差、原行/token身份、p匹配/unknown、CPU/内存benchmark | MUST | PLANNED / NOT_CLAIMED | config/tokenizer_config SHA现已匹配原capture；重放及native p身份仍未验证；不用block15 logit lens冒充p |
| PLAN-RB-M2 | M2 | 条件→prevalue事实所属主体剖面；A1概率/A2关注/A3归因/A4持续分别计量 | M0许可QA与全900库存，source-only构造角色交换/保义重排；其他任务按实际量分开 | 关系响应、顺序稳定、同值/范围控制、32-head全记录、pre/post区别、coverage | MUST | PLANNED / NOT_CLAIMED | M0/M1通过；分析与聚合先冻结，无目标标签挑头 |
| PLAN-RB-M3 | M3 | 固定值/位置的Q-only/K-only/V-only角色路径，no-op/无关/等范数sham | 独立来源控制，必要时原冻结Llama新forward；不只补采FN | 最终原token效应、角色交互、正确内容损伤、sham比较、实际forward/token/GPU秒 | MUST / CONDITIONAL | PLANNED / NOT_CLAIMED | M2给出不同预测；原模型/后端/资源门；新科学准入而非仅指纹新 |
| PLAN-RB-M4 | M4 | 原回答检测/定位/成本：内部候选 vs 固定强文本NLI vs R4/V3及p/attention消融 | 原900 QA全分母起步；六任务推广另有同合同准入，不冒充已覆盖 | AUROC/AP、固定预算FN/FP、span定位、coverage/unknown、source成组区间、完整及增量成本 | MUST / CONDITIONAL | PLANNED / NOT_CLAIMED | 完整detector/unknown/尺度/阈值/分数freeze；评价最后开真值 |

## 当前顺序

1. 首先只进入M0的source-only原人口库存与M1合同设计审查；两者都不需要新增模型推理。
2. M1实测后填写CPU吞吐和全量时长；M2复用其关系profiles，避免无理由重复读大缓存。真实source角色交换若无同合同cache，必须另行准入采集，不能仅重标来源角色就声称模型有响应。
3. M3只有在独立角色路径假设可区分解释且有精确模型资源时才claim；未满足则保持PLANNED。GPU小时没有可信实测值，不填造值。
4. M4必须在新检测器通过准入并冻结全部输出后开始。不能用M2热图、M3局部效应或玩具问答正确率代替检测结果。

## 停止与保留

- 来源许可不足保留unknown，先判定前端问题；不强制双射，不把正确算术/多跳当错绑。
- 仅高p/高attention/集中归因/稳定而无关系特异证据，不晋升方法；owner-switch不保证覆盖值/读出失败。
- 目标标签不用于选head、层、方向、阈值、prompt或模块；构造语法控制不训练探针、不按正确率挑最好头。
- 对旧P002/P003/P005/P021/P026和H004/H005/H012/H015/H016引用既有结论；换名、换距离或换世界权重不构成新run准入。
- 先科学准入，再代码/输入/观察器身份与重复检查，最后实际claim。失败与未覆盖不删除；本计划不安装自动GPU队列。

## 当前费用与结果

| 项目 | 当前实际值/状态 |
|---|---|
| 本计划新增模型调用 | 0 |
| 本计划新目标标签读取 | 0 |
| 新检测评分/自然AUROC | 无 |
| 新检测器科学准入 | 未完成 |
| M0–M2新增GPU预算 | 计划0；不含未来明确准入的新控制采集 |
| M3/M4 GPU小时 | 待原模型身份、资源门和实际吞吐benchmark；未估算 |
| 主收益与非劣界 | run claim前固定；尚未设定，不能看标签结果后补定 |
| 下一项需要记录 | M0/M1设计审查、明确输入/输出/unknown与新的区分信息 |


主代理审查更新：已知错例仅作研究动机；M2观测准入与M2/M3联合补采的最终验收分开。读取契约纠错保留旧版本；没有新增模型运行或评分。
