# 性能优先任务表

日期2026-11-02。允许多模块，目标各主任务AUROC≥0.8；真实标签仅冻结后的最终评价使用。

| ID | 任务 | 状态 | 运行条件／已有证据 |
|---|---|---|---|
| S001 | R4/V3六主任务与Hallu三组统计 | RECOVERED_NOT_RERUN | R4完整Hallu TEST已恢复；旧DEV132单列 |
| S002 | 真实FN/FP归类 | RECOVERED_NOT_RERUN | 原稿／历史人工标注与执行输出 |
| P001 | 恢复成对token分数／观测 | WAITING_DATA_ACCESS | 当前SSH不可达、完整缓存不在本地 |
| P002 | 固定等权秩融合CPU验证 | TODO_DEPENDS_P001 | 必须有同轴数组，不能由AUROC表推算 |
| P003 | 冻结基础模型核验器合同／采集 | TODO_RESOURCE_AND_CONTRACT | 不要求关系反事实正确性，判断允许unknown |
| P004 | 完整评分封存、最终效果与消融 | TODO | R4、V3、固定融合、核验单独、融合＋核验 |
| U001–U003 | 四条件操作／预测诊断 | OPTIONAL_SUPPORTING | 可辅助模块设计，不作为唯一允许路线 |
| R001–R008 | 旧关系组合 | DEFERRED_RELATION_ASSUMPTION_UNRESOLVED | 仅闭合语义合同下可选 |

目标完成数：R4点值2/6，V3点值3/6；最好点值包络4/6并非新方法。全部达标尚未实现，没有新效能结果。
