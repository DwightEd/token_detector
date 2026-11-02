# 机制研究执行表

2026-11-03。研究继续推进；长期 goal 控制器仍显示旧 blocked 状态。状态反映实际执行；合同测试通过不等于真实数据有效。

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
| H004-Q1 | 来源类型表达×规则提醒交叉对照 | COMPLETED_NOT_PROMOTED | 34×4×6逻辑请求，240新增前向；仅3自然来源转换，构造0覆盖；docs/results/h004_typed_factorial.md |
| H005-Q1 | 完整回答与精确单元焦点 | COMPLETED_NOT_PROMOTED | 原34答、104单元，714新增前向；字符/句子轴混合效果，不是历史BPE；docs/results/h005_focused_verdict.md |
| H005-D1 | 固定单元的标签密度排序上界 | POSTHOC_DIAGNOSTIC_ONLY | 13自然答字符轴，绝不用于预测；大部分排序损失不能只归咎于分段；outputs/h005-20261103/unit-ceiling-audit.json |
| H006-Q1 | 冻结外部监督NLI诊断 | COMPLETED_NOT_PROMOTED | 34答104单元、144来源文本对、18批前向；标准对照而非主方法；表格交换/合法计算仍错；docs/results/h006_external_nli.md |
| H007-Q1 | 关系方向×回答判定负担新控制 | PREPARING_NOT_RUN | 既定NEXT_MECHANISM第2比较；来源固定且全部保留，0/4/12正确干扰句，matched whole/target；不据自然gold选模板 |
| H001-Q0 | 前导 ASCII 空格的同回答配对控制 | CONTRACT_ONLY_NOT_RUN | 映射测试不等于观测机制已识别；旧生成器混杂仍未解除 |
| H002-D1 | 比较器、全token投影、unknown与评分冻结 | NOT_DEFINED_PENDING_QUALIFICATION | 不临时加权，不拿观察输出直接称最终检测器 |
| H002-E1 | 完整六任务与模块消融 | NOT_RUN | 原人口/原标签；每项 AUROC/AP/区间与成本 |
| H002-E2 | 未接触确认与新颖性审查 | NOT_RUN | 资格案例与历史已看人群不可改名为确认集 |

恢复入口：[RESUME](../research-state/RESUME.md)。下一问题：在来源不变、目标关系方向全因子平衡时，增加回答中的正确无关命题会不会改变正确/错误关系的分离，精确焦点是否保留该分离？H002 核心值盲重建仍未检验，但不再重复失败的自由索引前端。
