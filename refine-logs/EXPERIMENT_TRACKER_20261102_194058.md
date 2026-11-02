# 机制研究执行表

2026-11-02。goal active。状态反映实际执行；合同测试通过不等于真实数据有效。

| ID | 目的/当前动作 | 状态 | 证据与门禁 |
|---|---|---|---|
| K001 | R4/V3、固定等权与真实 FN/FP 审计 | COMPLETED_REUSE | docs/results 与 prior_experiments 复用，不重跑 |
| G001 | 数据/观察合同与先前覆盖审查 | COMPLETED | research-state/dataset_contracts.json；31 条先前实验索引 |
| G002 | 持久目标、上下文恢复与精确运行去重 | IMPLEMENTED_TESTED | tools/research_memory.py；22 项合同检查已过；不冒充外部 watchdog |
| G003 | 固定完整自然面板、评价隔离 | PREPARED_NOT_SCORED | 18 答，case_panel_v1.json 中输入/评价材料分别绑定 |
| G004 | H002 自动提议与 direct/blind/leaky | FROZEN_REVIEWED_INSTALLED_CHECKS_PASSED | 94安装模块身份一致，78合成检查通过；无效/未知/上限全部保留 |
| G005 | 独立 CUDA 环境及新观察模型 | PREPARED_KERNEL_PROBE_PASSED | 9模型文件匹配官方快照，固定依赖/实际CUDA与NF4微型运算通过；未证明完整3B装载已成功 |
| H002-Q1 | 18 答自动语义资格 | IDLE_JOB_ARMED | 19:39真实任务已启动，等内存准入；实时状态见outputs/research-goal-20261102/autostart-state.json；当时零LM调用 |
| H002-Q2 | 同词袋归属/证据缺失/合法推导控制 | PREPARED_SEALED_NOT_RUN | 16答/8对，六组严格词袋相同，独立审查通过；仍待Q1后决定执行 |
| H001-Q0 | 前导 ASCII 空格的同回答配对控制 | CONTRACT_ONLY_NOT_RUN | 映射测试不等于观测机制已识别；旧生成器混杂仍未解除 |
| H002-D1 | 比较器、全token投影、unknown与评分冻结 | NOT_DEFINED_PENDING_QUALIFICATION | 不临时加权，不拿观察输出直接称最终检测器 |
| H002-E1 | 完整六任务与模块消融 | NOT_RUN | 原人口/原标签；每项 AUROC/AP/区间与成本 |
| H002-E2 | 未接触确认与新颖性审查 | NOT_RUN | 资格案例与历史已看人群不可改名为确认集 |

恢复入口：[RESUME](../research-state/RESUME.md)。下一问题：自动、无人工错误位置的值盲来源重建是否有足够覆盖，能否与同预算直接核验形成可解释差异？
