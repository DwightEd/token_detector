# 机制研究执行表

2026-11-02。goal active。状态反映实际执行；合同测试通过不等于真实数据有效。

| ID | 目的/当前动作 | 状态 | 证据与门禁 |
|---|---|---|---|
| K001 | R4/V3、固定等权与真实 FN/FP 审计 | COMPLETED_REUSE | docs/results 与 prior_experiments 复用，不重跑 |
| G001 | 数据/观察合同与先前覆盖审查 | COMPLETED | research-state/dataset_contracts.json；31 条先前实验索引 |
| G002 | 持久目标、上下文恢复与精确运行去重 | IMPLEMENTED_TESTED | tools/research_memory.py；22 项合同检查已过；不冒充外部 watchdog |
| G003 | 固定完整自然面板、评价隔离 | H002_OBSERVATIONS_AND_CPU_REVIEW_SEALED | 18/18保留；H003复用完整输入，分数封存后才释放评价材料 |
| G004 | H002 自动提议与 direct/blind/leaky | PROPOSAL_FAILED_CORE_UNTESTED | 95文件冻结版本；18次提议、0slot，来源三路各0调用；不修补历史输出 |
| G005 | 独立 CUDA 环境及新观察模型 | ACTUAL_MODEL_LOAD_AND_INFERENCE_PASSED | 9文件匹配官方快照；完整3B已完成H002，H003也已加载；仅证明工程可运行 |
| H002-Q1 | 18 答自动语义资格 | COMPLETED_FRONTEND_FAILURE | 48frame，8合法均空slot，40无效/4envelope失败；约793秒，核心机制未检验 |
| H003-Q1 | 无自由提取的整回答语义基线 | COMPLETED_NOT_PROMOTED | 18/108前向，41.48秒；全18有排列变化，自然错误仍有高支持分；CPU复核与助手盲审计完成，不是token AUROC |
| H003-Q2 | 同词袋归属/证据缺失/合法推导控制 | COMPLETED_NOT_NATURAL_EFFICACY | 16/96前向12.58秒；三类11/16，方向6/7；null/False反向，正确计算未识别；旧H002接口政策不当构造真值 |
| H004-Q1 | 来源类型表达×规则提醒交叉对照 | PROPOSED_NOT_IMPLEMENTED | 先检验字段/类型保留，四条件隔离模块作用；不改权重、不混入算术执行 |
| H001-Q0 | 前导 ASCII 空格的同回答配对控制 | CONTRACT_ONLY_NOT_RUN | 映射测试不等于观测机制已识别；旧生成器混杂仍未解除 |
| H002-D1 | 比较器、全token投影、unknown与评分冻结 | NOT_DEFINED_PENDING_QUALIFICATION | 不临时加权，不拿观察输出直接称最终检测器 |
| H002-E1 | 完整六任务与模块消融 | NOT_RUN | 原人口/原标签；每项 AUROC/AP/区间与成本 |
| H002-E2 | 未接触确认与新颖性审查 | NOT_RUN | 资格案例与历史已看人群不可改名为确认集 |

恢复入口：[RESUME](../research-state/RESUME.md)。下一问题：自动、无人工错误位置的值盲来源重建是否有足够覆盖，能否与同预算直接核验形成可解释差异？
