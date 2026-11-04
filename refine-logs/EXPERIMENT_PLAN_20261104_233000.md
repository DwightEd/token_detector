# H018：已完成原两回答，下一门是字段归属交换

2026-11-04 23:30，Asia/Shanghai。当前事实F056/F057、[公开P1结果](../docs/results/h018_local_source_use.md)及[恢复状态](../research-state/RESUME.md)。六任务各0.8未达成，未建立新的有效检测器。

P1实际完成2native/8head/256extraNorm+MLP，407原token分数封存；独立审计全部通过、0目标GT。合法not73主e≈−0.00520而sham≈−0.00599，没有关系特异性；not65数值unresolved、lot66弱OR unknown。完整原R/raw直接复用，不重跑。

唯一下一问题：保持R5684、原风险、全部32层/allsource，交换字段归属后读出是否跟随回答已指名字段，而不随无关字段变化。详细预言/约束见[第二门提案](operator-binding-20261104/PHASE2_PROPOSAL_20261104_232000.md)；独立审查6f2552c9…仅允许**零模型前端及编排实现**，没有模型执行许可或armed预算。

| 顺序 | 对象/成本 | 冻结门与结论 |
|---|---|---|
| 1 输入构造 | 新S_target=valetFalse↔OutdoorTrue；S_control=garageFalse↔OutdoorTrue；0模型 | 唯一typed spans、完整R不变、street保持；来源token Counter相同/恰2bool位置/P803/source707/T233/N1036，否则stop |
| 2 复用与代码 | 原S0 case5684旧seal；0重跑；新编排复用原observe_case/kernel/capture | 数值依赖SHA/AST保持，独立fresh review及installed checks；不得恢复旧API别名或复制方法代码 |
| 3 有限新模型 | **尚未准入/未启动**；拟2新native/8head/256extraMLP | 必须另资源审核、原数据冻结及exact claim；原FP32/512MiB/6GiB门保持；完整233×2风险一次封存 |
| 4 无标签判别 | 仅成功后审计，0额外模型 | 原not73 sign与same-q875 not539/available2561辅助margin、P006/NLL/sham全报；不代替主risk或自然AUROC；失败无调sign/layer/scope |

原R有共同的hours错误前缀，三世界都保留，不修R或做prefix reset；outdoor donor不在not73前词面出现。物理位置/原历史错误/Norm generic damage仍需保留，不能将输入变化包装成唯一电路因果或新颖性。即使通过也不自动准入900。

本机完整CPU请求达到300秒上限，记录incomplete_timeout；37新installed tests已通过。发布必须检查精确push提交的完整Linux CI，不能拿本机targeted或旧CI替代。所有旧失败与历史计划保留。
